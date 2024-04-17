import argparse
import numpy as np

import torch
import torch.optim as optim

from PIL import Image
import tqdm

from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterXL

#后面记得改成parser.add_argument
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"


def style(self, iters=500):
    if iters > 0:
        
        #self.prepare_train() 被放到main里面了 dream gaussian都封装在gui里面了
        for i in tqdm.trange(iters):
            i = i #纯占位
            #train_step()
        # save
    #save_model()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/mnt/d/dataset/data_DTU/dtu_scan105/',
                        help='input data directory')
    #init Stable
    pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    
    #init IPAdapterXL
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

    #IPAdapter demo中 图片放这里
    #应该要改成 Gaussian 给图 
    #可以学一下 dream gaussian封装一个函数
    image = "./assets/0.jpg"
    image = Image.open(image)
    image.resize((512, 512))
    # images = ip_model.generate 所以要去IPAdapterXL里面增加一个refine函数

    # style_image 


    