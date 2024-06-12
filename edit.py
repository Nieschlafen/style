import os
import torch
import time
import math
import torchvision
import numpy as np
import ui_utils
import viser
import viser.transforms as tf
import random
from PIL import Image
from tqdm import tqdm
from threestudio.utils.typing import *  # Dict
from omegaconf import OmegaConf
from argparse import ArgumentParser
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.cameras import Simple_Camera
from gaussiansplatting.scene.camera_scene import CamScene
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.misc import get_device
from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from initialize.EditGuidance import EditGuidance
from initialize.CustomGuidance import CustomGuidance
from gaussiansplatting.gaussian_renderer import render
from arguments import ModelParams


class editpara:
    def __init__(self, cfg) -> None:
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.port = 8084
        # training cfg
        self.use_sam = False
        self.guidance = None
        self.stop_training = False
        self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        # from original system
        self.points3d = []
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.resolution_slider = 2048
        self.FoV_slider = 1
        self.renderer_output = "comp_rgb"
        edit_save = "edit_save1"
        self.model_path = os.path.join(".", edit_save, "pth")
        os.makedirs(self.model_path, exist_ok=True)

        self.show_semantic_mask = False
        # parameter
        self.anchor_weight_init_g0 = 0.05
        self.anchor_weight_init = 0.1
        self.anchor_weight_multiplier = 1.3
        self.edit_train_steps = 30
        self.gs_lr_scaler = 3.0
        self.gs_lr_end_scaler = 2.0
        self.color_lr_scaler = 3.0
        self.opacity_lr_scaler = 2.0
        self.scaling_lr_scaler = 2.0
        self.rotation_lr_scaler = 2.0
        self.sam_enabled = False
        self.edit_cam_num = 1  # cam number
        # guidance parameter
        self.per_editing_step = 10
        self.edit_begin_step = 0
        self.edit_until_step = 1000
        self.lambda_l1 = 10
        self.lambda_p = 10
        self.lambda_anchor_color = 0
        self.lambda_anchor_geo = 50
        self.lambda_anchor_scale = 50
        self.lambda_anchor_opacity = 50
        self.densify_until_step = 1300
        self.densification_interval = 100
        self.max_densify_percent = 0.01
        self.min_opacity = 0.05

        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        # diffusion model
        self.ip2p = None
        self.stn_ip2p = None
        self.ctn = None

        self.ctn_inpaint = None
        self.training = False
        if self.colmap_dir is not None:
            # Reading camera N/N
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.masks_2D = {}

        self.epoch = 0
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())
        self.sam_predictor = self.text_segmentor.model.sam
        self.sam_predictor.is_image_set = True
        self.sam_features = {}
        self.semantic_gauassian_masks = {}
        self.semantic_gauassian_masks["ALL"] = torch.ones_like(self.gaussian._opacity)

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.server = viser.ViserServer(port=self.port)

        with self.server.add_gui_folder("Editor"):
            self.edit_type = self.server.add_gui_dropdown(
                "Edit Type", ("Edit",)
            )
            self.guidance_type = self.server.add_gui_dropdown(
                "Guidance Type", ("InstructPix2Pix", "instantstyle", "custom")
            )
            self.edit_frame_show = self.server.add_gui_checkbox(
                "Show Edit Frame", initial_value=True, visible=False
            )
            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )
            self.edit_text = self.server.add_gui_text(
                "Text",
                initial_value="",
                visible=True,
            )
            self.edit_begin_button = self.server.add_gui_button("Edit Begin!")
            self.edit_end_button = self.server.add_gui_button(
                "End Editing!", visible=False
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

        @self.save_button.on_click
        def _(_):
            save_path = os.path.join(edit_save)
            self.gaussian.save_ply(os.path.join(edit_save, "pth", "point_cloud", "iteration_1500", "point_cloud.ply"))
            print("\n[ITER {}] Saving Checkpoint".format(self.edit_train_steps))
            torch.save(self.gaussian.capture(), self.model_path + "/chkpnt" + "test.pth")
            print(f"文件已保存到 {save_path}")

        @self.edit_end_button.on_click
        def _(event: viser.GuiEvent):
            self.stop_training = True

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

        @self.edit_begin_button.on_click
        def _(event: viser.GuiEvent):
            self.edit_begin_button.visible = False
            self.edit_end_button.visible = True
            if self.training:
                return
            self.training = True
            self.configure_optimizers()  # 配置优化器
            self.gaussian.update_anchor_term(
                anchor_weight_init_g0=self.anchor_weight_init_g0,
                anchor_weight_init=self.anchor_weight_init,
                anchor_weight_multiplier=self.anchor_weight_multiplier,
            )
            self.edit_frame_show.visible = True
            edit_cameras, train_frames, train_frustums = ui_utils.sample_train_camera(self.colmap_cameras,
                                                                                      self.edit_cam_num,
                                                                                      self.server)
            if self.edit_type.value == "Edit":
                self.edit(edit_cameras, train_frames, train_frustums)
                ui_utils.remove_all(train_frames)
                ui_utils.remove_all(train_frustums)
                self.edit_frame_show.visible = False

            self.guidance = None
            self.training = False
            self.gaussian.anchor_postfix()
            self.edit_begin_button.visible = True
            self.edit_end_button.visible = False

        @self.edit_frame_show.on_update
        def _(_):
            if self.guidance is not None:
                for _ in self.guidance.train_frames:
                    _.visible = self.edit_frame_show.value
                for _ in self.guidance.train_frustums:
                    _.visible = self.edit_frame_show.value
                self.guidance.visible = self.edit_frame_show.value
        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser=ArgumentParser(description="Training script parameters"),
            max_steps=self.edit_train_steps,
            lr_scaler=self.gs_lr_scaler,
            lr_final_scaler=self.gs_lr_end_scaler,
            color_lr_scaler=self.color_lr_scaler,
            opacity_lr_scaler=self.opacity_lr_scaler,
            scaling_lr_scaler=self.scaling_lr_scaler,
            rotation_lr_scaler=self.rotation_lr_scaler,

        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)  # torch.optim.Adam

    def edit(self, edit_cameras, train_frames, train_frustums):
        if self.guidance_type.value == "InstructPix2Pix":
            if not self.ip2p:
                from threestudio.models.guidance.instructpix2pix_guidance import (
                    InstructPix2PixGuidance,
                )

                self.ip2p = InstructPix2PixGuidance(
                    OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
                )
            cur_2D_guidance = self.ip2p
            print("using InstructPix2Pix!")
        elif self.guidance_type.value == "instantstyle":
            if not self.stn_ip2p:
                from threestudio.models.guidance.instantstyle_guidance2 import (
                        newInstantStyleGuidance,
                )

                self.stn_ip2p = newInstantStyleGuidance(
                    OmegaConf.create({"min_step_percent": 0.05,
                                      "max_step_percent": 0.8,
                                     })
                )
            cur_2D_guidance = self.stn_ip2p
            print("using instantstyle!")
        elif self.guidance_type.value == "custom":
            if not self.ctn:
                from threestudio.models.guidance.custom_guidance import (
                        CustomStyleGuidance,
                )

                self.ctn = CustomStyleGuidance(
                    OmegaConf.create({"min_step_percent": 0.02,
                                      "max_step_percent": 0.98,
                                      "text_prompt": self.edit_text.value,
                                     })
                )
            cur_2D_guidance = self.ctn
            print("using customstyle!")

        origin_frames = self.render_cameras_list(edit_cameras)  # 原始帧
        # * EditGuidance and cur_2D_guidance *
        self.guidance = CustomGuidance(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
            origin_frames=origin_frames,
            text_prompt=self.edit_text.value,
            per_editing_step=self.per_editing_step,
            edit_begin_step=self.edit_begin_step,
            edit_until_step=self.edit_until_step,
            lambda_l1=self.lambda_l1,
            lambda_p=self.lambda_p,
            lambda_anchor_color=self.lambda_anchor_color,
            lambda_anchor_geo=self.lambda_anchor_geo,
            lambda_anchor_scale=self.lambda_anchor_scale,
            lambda_anchor_opacity=self.lambda_anchor_opacity,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
        )
        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]  # 1 H W C
            self.save_rendering(rendering, view_index)
            loss = self.guidance(rendering, view_index, step)
            # print("loss_sds:"+str(loss.item()))
            # self.densify_and_prune(step)
            loss.backward()
            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)

            if self.stop_training:  # end train
                self.stop_training = False
                return

    def save_rendering(self, rendering, view_index):
        # Assuming rendering is a tensor with shape (1, H, W, C)
        rendering = rendering.squeeze(0)  # Remove the batch dimension, new shape (H, W, C)
        rendering = rendering.detach().cpu().numpy()  # Convert to numpy array if it's a tensor

        # Convert rendering to uint8 if necessary (assuming rendering is in the range [0, 1])
        rendering = (rendering * 255).astype(np.uint8)

        # Convert to Image object and save
        image = Image.fromarray(rendering)
        output_dir = "./renderings"
        image_path = os.path.join(output_dir, f"rendering_{view_index}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames

    def render(
            self,
            cam,
            local=False,
            sam=False,
            train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        semantic_map = render(
            cam,
            self.gaussian,
            self.pipe,
            self.background_tensor,
            override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
        )["render"]
        semantic_map = torch.norm(semantic_map, dim=0)
        semantic_map = semantic_map > 0.0  # 1, H, W
        semantic_map_viz = image.detach().clone()  # C, H, W
        semantic_map_viz = semantic_map_viz.permute(1, 2, 0)  # 3 512 512 to 512 512 3
        semantic_map_viz[semantic_map] = 0.50 * semantic_map_viz[
            semantic_map
        ] + 0.50 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
        semantic_map_viz = semantic_map_viz.permute(2, 0, 1)  # 512 512 3 to 3 512 512

        render_pkg["sam_masks"] = []
        render_pkg["point2ds"] = []
        if sam:
            if hasattr(self, "points3d") and len(self.points3d) > 0:
                sam_output = self.sam_predict(image, cam)
                if sam_output is not None:
                    render_pkg["sam_masks"].append(sam_output[0])
                    render_pkg["point2ds"].append(sam_output[1])

        self.gaussian.localize = False  # reverse

        render_pkg["semantic"] = semantic_map_viz[None]
        render_pkg["masks"] = semantic_map[None]  # 1, 1, H, W

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    def densify_and_prune(self, step):
        if step <= self.densify_until_step:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent,
                    min_opacity=self.min_opacity,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output
        out_img = output[out_key][0]  # H W C
        if out_key == "comp_rgb":
            if self.show_semantic_mask:
                out_img = output["semantic"][0].moveaxis(0, -1)
        elif out_key == "masks":
            out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        if self.sam_enabled:
            if "sam_masks" in output and len(output["sam_masks"]) > 0:
                try:
                    out_img = torchvision.utils.draw_segmentation_masks(
                        out_img, output["sam_masks"][0]
                    )

                    out_img = torchvision.utils.draw_keypoints(
                        out_img,
                        output["point2ds"][0][None, ...],
                        colors="blue",
                        radius=5,
                    )
                except Exception as e:
                    print(e)

        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        output = self.render(gs_camera, sam=self.sam_enabled)
        out = self.prepare_output_image(output)

        self.server.set_background_image(out, format="jpeg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #

    args = parser.parse_args()
    edit = editpara(args)
    edit.render_loop()
