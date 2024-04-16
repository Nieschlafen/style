import torch
import os
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
from tqdm import tqdm

from ip_adapter import IPAdapterXL

base_model_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
image_encoder_path = "/data_heat/jcx/InstantStyle/sdxl_models/image_encoder"
ip_ckpt = "/data_heat/jcx/InstantStyle/sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

# 定义输入和输出文件夹路径
input_folder = "./input_images/"
output_folder = "./output_images/"
# 获取输入文件夹中的所有图像文件
os.makedirs(output_folder, exist_ok=True)
image_files = os.listdir(input_folder)

image = "./assets/4.jpg"
image = Image.open(image)
image = image.resize((512, 512))

mask_image3 = Image.open("./test_mask.png").convert("RGB")

num_images = len(image_files)

for image_file in tqdm(image_files, desc="Processing Images", total=num_images):
    image_path = os.path.join(input_folder, image_file)
    init_image = Image.open(image_path)
    # generate image
    images = ip_model.generate(pil_image=image,
                               # prompt="a dog sitting on, masterpiece, best quality, high quality",
                               negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                               scale=2.0,
                               guidance_scale=8,
                               num_samples=1,
                               # seed=42,
                               num_inference_steps=30,
                               image=init_image,
                               mask_image=mask_image3,
                               strength=0.99
                               )
    output_path = os.path.join(output_folder, image_file)
    images[0].save(output_path)

print("处理完成！")

