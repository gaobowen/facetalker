from diffusers import AutoencoderDC, SanaTransformer2DModel, UNet3DConditionModel
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import lpips
import cv2

import os
import glob
import argparse
import itertools
import math
import random
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_sana():
    # https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana.py
    # latent 320x320 -> 32x10x10
    num_frames = 5
    config = {
        "sample_size": 10,
        "in_channels": 32 * (num_frames + 1),
        "out_channels": 32 * num_frames,
        "num_layers": 12,
        "num_attention_heads": 60,
        "attention_head_dim": 32,
        "num_cross_attention_heads": 10,
        "cross_attention_head_dim":192,
        "cross_attention_dim": 1920,
        "caption_channels": 384,
    }

    model = SanaTransformer2DModel(**config).to(device)

    batch_size = 12
    in_channels = config["in_channels"]
    sample_size = config["sample_size"]
    caption_channels = config["caption_channels"]



    input_img = torch.randn((batch_size, in_channels, sample_size, sample_size)).to(device)
    # (batch, sequence_length, feature_dim)
    audio_feats = torch.randn((batch_size, 50, caption_channels)).to(device)

    timesteps =  torch.zeros(batch_size).to(device)

    for i in range(2000):
        image_pred = model(hidden_states=input_img, timestep=timesteps, encoder_hidden_states=audio_feats).sample
        if i % 100 == 0:
            print('image_pred', image_pred.shape)


def test_sana_hubert():
    # https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana.py
    # latent 384x384 -> 32x12x12
    num_frames = 5
    config = {
        "sample_size": 12,
        "in_channels": 32 * (num_frames + 1),
        "out_channels": 32 * num_frames,
        "num_layers": 12,
        "mlp_ratio": 4,
        "num_attention_heads": 32,
        "attention_head_dim": 32,
        "num_cross_attention_heads": 8,
        "cross_attention_head_dim":128,
        "cross_attention_dim": 1024,
        "caption_channels": 1024
    }

    model = SanaTransformer2DModel(**config).to(device)
    model.enable_gradient_checkpointing()
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)


    params_to_optimize = (
        itertools.chain(model.parameters()))
    
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=1e-4)

    batch_size = 2
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    sample_size = config["sample_size"]
    caption_channels = config["caption_channels"]


    dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()
    # dc_ae = torch.compile(dc_ae)
    dc_ae.requires_grad_(False)
    lpips_func = lpips.LPIPS(net='vgg').to(device).eval()
    lpips_func.requires_grad_(False)


    input_vae = torch.randn((batch_size, in_channels, sample_size, sample_size)).to(device)
    output_vae = torch.randn((batch_size, out_channels, sample_size, sample_size)).to(device)
    # (batch, sequence_length, feature_dim)
    audio_feats = torch.randn((batch_size, 9, caption_channels)).to(device)

    timesteps =  torch.zeros(batch_size).to(device)

    for i in range(2000):
        vae_pred = model(hidden_states=input_vae, timestep=timesteps, encoder_hidden_states=audio_feats).sample
        loss = F.mse_loss(vae_pred.float(), output_vae.float(), reduction="mean")
        vae_pred = vae_pred.half().reshape(-1, 32, 12, 12)

        # 太耗显存，拆分batch
        imgs_pred = []
        for b in range(0, vae_pred.shape[0]):
            imgs_pred.append(dc_ae.decode(vae_pred[b:b+1], return_dict=False)[0])
            torch.cuda.empty_cache()
        imgs_pred = torch.cat(imgs_pred, dim=0)
        lpips_func(imgs_pred, imgs_pred)
        loss.backward()
        # 参数更新
        optimizer.step()  
        optimizer.zero_grad()
        if i % 10 == 0:
            print('vae_pred', vae_pred.shape)


def test_unet3d():
    # latent 3x384x384 -> 32x12x12
    # sample shape  (batch, num_channels, num_frames, height, width)
    vae_size = 12
    num_frames = 5
    batch_size = 10
    in_channels = 32
    config = {
        "sample_size": 12,
        "in_channels": 32,
        "out_channels": 32,
        "cross_attention_dim": 2048,
    }
    
    model = UNet3DConditionModel(**config).to(device)

    input_vae = torch.randn((batch_size, config['in_channels'], num_frames+1, vae_size, vae_size)).to(device)

    audio_feats = torch.randn((batch_size, num_frames+4, config['cross_attention_dim'])).to(device)

    timesteps = torch.zeros(batch_size).to(device)
    for i in range(2000):
        vae_pred = model(input_vae, timesteps, encoder_hidden_states=audio_feats).sample
        if i % 100 == 0: print(vae_pred.shape)

if __name__ == "__main__":
    test_sana_hubert()
    # test_unet3d()
