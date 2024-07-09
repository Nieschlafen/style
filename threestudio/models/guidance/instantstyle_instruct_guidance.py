import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils.import_utils import is_xformers_available
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from ip_adapter import IPAdapterXL
from PIL import Image
import torchvision.transforms as transforms


@threestudio.register("stable-diffusion-instantstyle-guidance")
class InstantStyleGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None

        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path: str = "/home/lk/desktop/style/sdxl_models/image_encoder"
        ip_ckpt: str = "/home/lk/desktop/style/sdxl_models/ip-adapter_sdxl.bin"
        control_type: str = "normal"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_sds: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading instantstyle ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "cache_dir": self.cfg.cache_dir,
        }
        controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
        controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False,
                                                     torch_dtype=self.weights_dtype).to(self.device)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.cfg.base_model_path,
            controlnet=controlnet,
            torch_dtype=self.weights_dtype,
            add_watermarker=False,
            **pipe_kwargs,
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.pipe.enable_vae_tiling()
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.controlnet = self.pipe.controlnet

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        self.ip_model = IPAdapterXL(self.pipe, self.cfg.image_encoder_path, self.cfg.ip_ckpt, self.device,
                                    target_blocks=["up_blocks.0.attentions.1"])
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"finish_Loaded instantstyle!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
            self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
            self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            view_index,
            canny_map,
            **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        assert batch_size == 1

        # style image
        image = "./assets/14.jpg"
        image = Image.open(image)
        image.resize((512, 512))

        """cond_rgb_numpy = cond_rgb.cpu().numpy()
        cond_rgb_numpy = (cond_rgb_numpy * 255).astype(np.uint8)
        retval, buffer = cv2.imencode('.jpg', cond_rgb_numpy[0])
        rbimage = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        detected_map = cv2.Canny(rbimage, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))"""
        # generate image
        images = self.ip_model.generate(pil_image=image,
                                        prompt="best quality, high quality",
                                        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                        scale=1.0,
                                        guidance_scale=5,
                                        num_samples=1,
                                        num_inference_steps=20,
                                        seed=42,
                                        image=canny_map,
                                        controlnet_conditioning_scale=0.6,
                                        )
        images[0].save(f"./renderings/result_{view_index}.png")
        image = images[0]
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 调整图像大小为 512x512
            transforms.ToTensor(),  # 转换为 PyTorch 张量并归一化到 [0, 1] 之间
        ])
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, 512, 512]
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        image_tensor = image_tensor.to(self.device)

        return {"edit_images": image_tensor}
