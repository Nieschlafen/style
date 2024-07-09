import sys
import inspect
from dataclasses import dataclass
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from PIL import Image
from safetensors import safe_open
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

from ip_adapter.utils import is_torch2_available, get_generator

if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from ip_adapter.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


@threestudio.register("stable-diffusion-newinstantstyle-guidance")
class newInstantStyleGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None

        ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        # image_encoder_path: str = "/home/yjx/desktop/InstantStyle/sdxl_models/image_encoder"
        image_encoder_path: str = "/home/lk/desktop/style/sdxl_models/image_encoder"
        # ip_ckpt: str = "/home/yjx/desktop/InstantStyle/sdxl_models/ip-adapter_sdxl.bin"
        ip_ckpt: str = "/home/lk/desktop/style/sdxl_models/ip-adapter_sdxl.bin"
        control_type: str = "normal"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        fixed_size: int = -1
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        lambda_sd: float = 0.01
        diffusion_steps: int = 30
        seed: int = 42
        use_sds: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading new instantstyle ...")

        self.weights_dtype = torch.float16

        pipe_kwargs = {"cache_dir": self.cfg.cache_dir, }
        controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
        controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False,
                                                     torch_dtype=self.weights_dtype).to(self.device)
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.cfg.base_model_path,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.weights_dtype,
            add_watermarker=False,
            **pipe_kwargs,
        ).to(self.device)
        self.target_blocks = ["up_blocks.0.attentions.1"]
        self.num_tokens = 4
        self.set_ip_adapter()

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )

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
        self.guidance_scale = 5.0
        self.ip_ckpt = self.cfg.ip_ckpt
        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.cfg.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        self.lambda_sd = 0.01
        self.do_classifier_free_guidance = True

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.generator = get_generator(self.cfg.seed, self.device)
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"finish_Loaded new instantstyle!")

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = False
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                    ).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        skip=True
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
            self,
            control_model_input,
            t,
            encoder_hidden_states,
            controlnet_cond,
            conditioning_scale,
            guess_mode,
            added_cond_kwargs,
            return_dict,
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            control_model_input.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=controlnet_cond.to(self.weights_dtype),
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
            self,
            latents_model_input,
            t,
            encoder_hidden_states,
            timestep_cond,
            cross_attention_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
            added_cond_kwargs,
            return_dict,
    ) -> Float[Tensor, "..."]:
        return self.unet(
            latents_model_input.to(self.weights_dtype),
            t,
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents.to(input_dtype)

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def prepare_prompt_embed(
            self,
            style_image,  # 风格图
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=42,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = 1
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        pooled_prompt_embeds_ = None
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(style_image,
                                                                                content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            global_step,
            canny_map,
            **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        assert batch_size == 1
        height: Optional[int] = None
        width: Optional[int] = None
        num_images_per_prompt: Optional[int] = 1
        guess_mode: bool = False
        original_size: Tuple[int, int] = None
        target_size: Tuple[int, int] = None
        crops_coords_top_left: Tuple[int, int] = (0, 0)
        num_inference_steps: int = 30
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0
        control_guidance_start: Union[float, List[float]] = 0.0
        control_guidance_end: Union[float, List[float]] = 1.0
        cross_attention_kwargs: Optional[Dict[str, Any]] = None

        pred_rgb = rgb.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        img_rgb = pred_rgb
        pred_rgb_512 = F.interpolate(img_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_512)

        """cond_rgb_numpy = rgb.cpu().detach().numpy()
        cond_rgb_numpy = (cond_rgb_numpy * 255).astype(np.uint8)
        # cv2.imwrite(f'cond_rgb_image_{view_index}.jpg', cv2.cvtColor(cond_rgb_numpy[0], cv2.COLOR_RGB2BGR))
        retval, buffer = cv2.imencode('.jpg', cond_rgb_numpy[0])
        rbimage = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        detected_map = cv2.Canny(rbimage, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))"""

        # style image
        image = "/home/lk/desktop/style/assets/4.jpg"
        image = Image.open(image)
        image.resize((512, 512))

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
        # 3.1 Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.prepare_prompt_embed(
            style_image=image,
            prompt="best quality, high quality",
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            scale=1.0,
            num_samples=1,
            num_inference_steps=30,
            seed=self.cfg.seed,
        )
        # 4. Prepare image
        image = self.prepare_image(
            image=canny_map,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=self.device,
            dtype=self.weights_dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        # 6. Prepare latent variables
        generator = get_generator(self.cfg.seed, self.device)
        num_channels_latents = self.unet.config.in_channels

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=latents.dtype)
        # 7. Prepare extra step kwargs.
        eta: float = 0.0
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = self._num_timesteps - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        cur_iters = global_step
        total_iters = 10000  # 10000
        max_step = self.max_step  # 980
        min_step = self.min_step  # 20
        if cur_iters > total_iters / 2:
            max_step = int(max_step * 0.5)
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        t_ratio = 1
        t = (t * t_ratio).to(torch.long)

        if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
            torch._inductor.cudagraph_mark_step_begin()

        with torch.no_grad():
            noise = torch.randn_like(latents)  # 1 4 64 64
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
        # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        # controlnet(s) inference
        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds
        controlnet_added_cond_kwargs = added_cond_kwargs
        cond_scale = 1.0

        down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            added_cond_kwargs=controlnet_added_cond_kwargs,
            return_dict=False,
        )
        # predict the noise residual
        noise_pred = self.forward_unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]  # 2 4 64 64
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)  # 1 4 64 64

        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise) * self.lambda_sd
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum")
        loss_dict_sd = dict(loss_sds=loss_sds.item())

        loss_dict = {}
        loss = loss_sds
        loss_dict.update(loss_dict_sd)

        return loss, loss_dict