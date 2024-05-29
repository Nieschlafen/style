from dataclasses import dataclass
import threestudio
import torch
import sys
import torch.nn.functional as F
from gs.c_clip import CLIP
import clip

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *
from threestudio.utils.misc import C, parse_version


@threestudio.register("stable-diffusion-customstyle-guidance")
class CustomStyleGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
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
        negative: str = ""
        text_prompt: str = ""

        diffusion_steps: int = 20
        num_tokens: int = 4

        use_sds: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading customstyle ...")

        self.weights_dtype = torch.float16

        # model_key = "/home/llq/Data/model_weight/runwayml/stable-diffusion-v1-5"
        model_key = "runwayml/stable-diffusion-v1-5"
        print(f"load model: {model_key}")
        self.pipe = DiffusionPipeline.from_pretrained(model_key).to(self.device)

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps  # 100
        self.clip_guidance = CLIP(self.device)
        prompt = ["front face of an object", "side face of an object", 'back face of an object']
        self.clip_match_text = clip.tokenize(prompt).to(self.device)  # torch.Size([3, 77])

        for p in self.clip_guidance.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.set_min_max_steps()  # set to default value
        self.prepare_text_embeddings()
        self.lambda_sd = 0.01

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = None
        self.null_dict = dict()

        threestudio.info(f"finish_Loaded customstyle!")

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
        class_labels = None
        return self.unet(latents, t, encoder_hidden_states=encoder_hidden_states, class_labels=class_labels).sample

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2 - 1
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

    def prepare_text_embeddings(self):
        if self.cfg.text_prompt is None:
            print(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        def get_sd_text_emb(input_text):
            text_z_i = self.get_text_embeds([input_text], [self.cfg.negative])
            print(f"Preparing text embedding for: {input_text}")
            return text_z_i

        self.text_z = get_sd_text_emb(self.cfg.text_prompt)

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.cuda.amp.autocast(enabled=False)
    def train_step(self, latents, text_embeddings, global_step, t_ratio=1):
        guidance_scale = 100
        cur_iters = global_step
        total_iters = 10000  # 10000
        max_step = self.max_step  # 980
        min_step = self.min_step  # 20
        if cur_iters > total_iters / 2:
            max_step = int(max_step * 0.5)
        # t = max(int((1 - cur_iters / total_iters) * max_step), min_step)
        # t = torch.tensor([t], dtype=torch.long, device=self.device)
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)

        t = (t * t_ratio).to(torch.long)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.forward_unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_uncond - noise_pred_text)

        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise) * self.lambda_sd

        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()  # ok
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum")
        loss_dict = dict(loss_sds=loss.item())

        return loss, loss_dict

    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            global_step,
            cond_rgb: Float[Tensor, "B H W C"],
            **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        pred_rgb = rgb.permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        img_rgb = pred_rgb
        pred_rgb_512 = F.interpolate(img_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_512)

        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (H, W),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_images(cond_rgb_BCHW_HW8)
        # combined_latents = torch.cat((latents, cond_latents), dim=0)
        t_ratio = 1
        """temp = torch.zeros(1).to(rgb.device)
        text_embeddings = prompt_guidance.get_text_embeddings(temp, temp, temp, False)"""
        match_probs = None
        with torch.no_grad():
            images = self.clip_guidance.transformCLIP(pred_rgb)
            logits_per_image, logits_per_text = self.clip_guidance.model(images, self.clip_match_text)
            match_probs = logits_per_image.softmax(dim=1)
        select = match_probs.max(-1).indices.tolist()
        # text_z = torch.cat([self.text_z[t].clone() for t in select])
        text_z = self.text_z

        loss_dict = {}
        loss_sd, loss_dict_sd = self.train_step(latents, text_z, global_step, t_ratio=t_ratio)
        loss = loss_sd
        loss_dict.update(loss_dict_sd)

        return loss, loss_dict