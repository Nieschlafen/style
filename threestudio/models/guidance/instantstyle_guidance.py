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


@threestudio.register("stable-diffusion-instantstyle-guidance")
class InstantStyleGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None

        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path: str = "/data_heat/jcx/InstantStyle/sdxl_models/image_encoder"
        ip_ckpt: str = "/data_heat/jcx/InstantStyle/sdxl_models/ip-adapter_sdxl.bin"
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

        self.weights_dtype = torch.float16

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
            ** pipe_kwargs,
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
        self.lambda_sd = 0.01

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
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

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ):
        guidance_scale = 100
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.forward_unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise) * self.lambda_sd
        return grad

    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            cond_rgb: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        assert batch_size == 1
        assert rgb.shape[:-1] == cond_rgb.shape[:-1]

        ip_model = IPAdapterXL(self.pipe, self.cfg.image_encoder_path, self.cfg.ip_ckpt, self.device, target_blocks=["up_blocks.0.attentions.1"])
        # style image
        image = "./assets/4.jpg"
        image = Image.open(image)
        image.resize((512, 512))

        cond_rgb_numpy = cond_rgb.cpu().numpy()
        cond_rgb_numpy = (cond_rgb_numpy * 255).astype(np.uint8)
        cv2.imwrite('cond_rgb_image.jpg', cv2.cvtColor(cond_rgb_numpy[0], cv2.COLOR_RGB2BGR))
        retval, buffer = cv2.imencode('.jpg', cond_rgb_numpy[0])
        rbimage = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        detected_map = cv2.Canny(rbimage, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

        output_path2 = "canny_output.png"
        canny_map.save(output_path2)
        # generate image
        images = ip_model.generate(pil_image=image,
                                   prompt="best quality, high quality",
                                   negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                   scale=1.0,
                                   guidance_scale=5,
                                   num_samples=1,
                                   num_inference_steps=30,
                                   seed=42,
                                   image=canny_map,
                                   controlnet_conditioning_scale=0.6,
                                   )
        images[0].save("result.png")
        image_array = np.array(images[0])
        image_tensor = torch.tensor(image_array, dtype=self.weights_dtype)
        image_tensor = image_tensor.unsqueeze(0)

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        print(f"rgb_BCHW_HW8 device before encoding: {rgb_BCHW_HW8.device}")
        latents = self.encode_images(rgb_BCHW_HW8)

        image_cond = image_tensor.permute(0, 3, 1, 2)
        image_cond = image_cond.to(torch.float32)
        image_cond = F.interpolate(
            image_cond, (RH, RW), mode="bilinear", align_corners=False
        )
        image_cond.to(rgb_BCHW_HW8.device)
        print(f"image_cond device before encoding: {image_cond.device}")
        cond_latents = self.encode_cond_images(image_cond)

        temp = torch.zeros(1).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step,self.max_step + 1,[batch_size],dtype=torch.long,device=self.device)

        grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
