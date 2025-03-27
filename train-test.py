from diffusers import AutoencoderDC, SanaTransformer2DModel
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import os
import argparse
import itertools
import math
import random
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


