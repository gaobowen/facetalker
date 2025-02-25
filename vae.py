
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers import AutoencoderDC
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm

device = torch.device("cuda")

dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()


resizeHW = 384

transform_normalize = transforms.Compose([ # -1 ~ 1
            transforms.Resize((resizeHW, resizeHW)),  # 调整图片大小
            transforms.ToTensor(),          # 转换为张量 0-1
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化[-1,1] Normalize = (tensor - mean) / std
        ])


if __name__ == "__main__":
    # mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
    
    # image = Image.open("./out.jpg")
    # x = transform_normalize(image)[None].to(device)
    # latent = dc_ae.encode(x, return_dict=False)[0]
    # print(latent.shape)
    # y = dc_ae.decode(latent, return_dict=False)[0]
    # save_image(y * 0.5 + 0.5, "demo_dc_ae.jpg")

    
    print('ok')
    
    # PyTorch scaled dot-product attention (SDPA). 
    # This attention implementation is activated by default for PyTorch versions 2.1.1 or greater.