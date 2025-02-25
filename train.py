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




logger = get_logger(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# latent 384x384 -> 32x12x12
num_frames = 5
sample_size = 12
in_channels = 32 * (num_frames + 1)
out_channels = 32 * num_frames
caption_channels = 384

config = {
    "sample_size": sample_size,
    "caption_channels": caption_channels,
    "in_channels": in_channels,
    "out_channels": out_channels,
}

model = SanaTransformer2DModel(**config).to(device)


batch_size = 1
input_img = torch.randn((batch_size, in_channels, sample_size, sample_size)).to(device)

# (batch, sequence_length, feature_dim)
audio_feats = torch.randn((batch_size, 50, caption_channels)).to(device)

timesteps = torch.tensor([0], device=device)

image_pred = model(hidden_states=input_img, timestep=timesteps, encoder_hidden_states=audio_feats).sample

print('image_pred', image_pred.shape)


