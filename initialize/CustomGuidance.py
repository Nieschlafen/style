import torch
import sys
from diffusers.image_processor import VaeImageProcessor
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
import ui_utils
# Diffusion model (cached) + prompts + edited_frames + training config

class CustomGuidance:
    def __init__(self, guidance, gaussian, origin_frames, text_prompt, per_editing_step, edit_begin_step,
                 edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
                 lambda_anchor_opacity, train_frames, train_frustums, cams, server
                 ):
        self.guidance = guidance
        self.gaussian = gaussian
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.origin_frames = origin_frames
        self.cams = cams
        self.server = server
        self.train_frames = train_frames
        self.train_frustums = train_frustums
        self.edit_frames = {}
        self.visible = True
        self.global_step = 0
        """self.prompt_guidance = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "/home/llq/Data/model_weight/runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )()"""
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())

    def __call__(self, rendering, canny_map, view_index, step):
        self.gaussian.update_learning_rate(step)
        self.global_step += 1
        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            result = self.guidance(
                rendering,
                view_index,
                canny_map,
            )
            self.edit_frames[view_index] = result["edit_images"].detach().clone() # 1 H W C
            self.train_frustums[view_index].remove()
            self.train_frustums[view_index] = ui_utils.new_frustums(view_index, self.train_frames[view_index],
                                                                    self.cams[view_index], self.edit_frames[view_index], self.visible, self.server)
        gt_image = self.edit_frames[view_index]

        loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
               self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                    gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()
        # anchor loss
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        if step % self.per_editing_step == 0:
            print("loss:" + str(loss.item()))

        return loss

