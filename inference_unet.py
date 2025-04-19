from diffusers import UNet2DConditionModel
import torch
import torch.nn as nn
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
from safetensors.torch import load_file

import os
import argparse
import itertools
import math
import random
from pathlib import Path
import json
from sd_vae import VAE

from utils.image_processor import ImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_processor = ImageProcessor(320, mask="fix_mask", device=str(device), )

vae = VAE(model_path="./models/sd-vae-ft-mse/", resized_img=320, use_float16=False, device=device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x


pe = PositionalEncoding(d_model=384)

transform = transforms.Compose([
                # transforms.Resize((resized_img, resized_img)),  # 调整图片大小
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化[-1,1] Normalize = (tensor - mean) / std   
            ])



def inference(step_num, test_dir):
    # raise OSError()

    with open('/data/gaobowen/facetalker/models/musetalk/musetalk.json', 'r') as f:
        unet_config = json.load(f)

    model = UNet2DConditionModel(**unet_config).to(device)
    # model.load_state_dict(torch.load("/data/gaobowen/facetalker/models/musetalk/unet.pth"))
    model.load_state_dict(load_file(f"/data/gaobowen/facetalker/output_unet2/checkpoint-{step_num}/model.safetensors"))

    model = model.to(device).eval()

    # AlexandriaOcasioCortez_1 AustinScott2_1 BarackObama_1 1733208337445-160927 
    # 1736236299779-160336
    # 1734340565511-115823 
    # 0feishu-20250416-171059 走动中国男性
    # 1734338622200-190107 方脸中国女性
    dataName = "1734338622200-190107"
    
    img_dir = f'/data/gaobowen/split_video_25fps_sdvae320-2/{dataName}'
    
    # img_dir = f'/data/gaobowen/vaildata_imgs/{dataName}'
    
    image_paths = glob.glob(os.path.join(img_dir, '*.jpg'), recursive=True)
    
    mp4_path = f"/data/gaobowen/split_video_25fps/{dataName}.mp4"
    
    # mp4_path = f"/data/gaobowen/vaildata/{dataName}.mp4"
    
    # filename = os.path.basename(imgpath)
    # img_idx = int(filename.replace('.jpg', ''))
    
    directory = img_dir
    
    audio_feats_npy = np.load(f'{test_dir}/output_whisper.npy')
    max_count, seqlen, dim = audio_feats_npy.shape
    print(audio_feats_npy.shape)

    # audio_feats_npy = np.load(f"{test_dir}/output_hu.npy")
    # max_count, dim = audio_feats_npy.shape
    # print(audio_feats_npy.shape)
    vae_hw = 40
    batchsize = 8
    # raise OSError()
    mode = "xI"
    max_img_idx = max_count - (max_count % batchsize)
    img_seq_len = len(mode) - 1
    
    def get_mp4_bg():
        if os.path.exists(f'{test_dir}/{dataName}/0.jpg'): return
        os.makedirs(f"{test_dir}/{dataName}", exist_ok=True)
        vide_capture = cv2.VideoCapture(mp4_path)
        index = 0
        while index <= max_img_idx + 1:
            ret, image = vide_capture.read()
            if ret:
                cv2.imwrite(f'{test_dir}/{dataName}/{index}.jpg', image)
                index += 1

    get_mp4_bg()

    img_idx = 0
    while img_idx < max_img_idx:
        # file_mode = "vae" # jpg vae
        def get_data(idx):
            ref_idx = idx
            input_vaes = []
            real_vaes = []
            real_imgs = []
            seq_idx = 0
            mode = "xI"
            for char in mode:
                if char == 'R':
                    ref_vae = torch.load(os.path.join(directory, f'{ref_idx}.vae')).reshape(-1, vae_hw, vae_hw)
                    input_vaes.append(ref_vae)
                elif char == 'I':
                    sel_vae = torch.load(os.path.join(directory, f'{idx}.vae')).reshape(-1, vae_hw, vae_hw)
                    input_vaes.append(sel_vae)
                    real_vaes.append(sel_vae)
                    seq_idx += 1
                elif char == 'x':
                    sel_vae = torch.load(os.path.join(directory, f'{idx}_mask.vae')).reshape(-1, vae_hw, vae_hw)
                    input_vaes.append(sel_vae)
                    real_vaes.append(torch.load(os.path.join(directory, f'{idx}.vae')).reshape(-1, vae_hw, vae_hw))
                    seq_idx += 1
                # print(img_idx + seq_idx)
            input_vaes_tensor =  torch.cat(input_vaes, dim=0) # (6*32, 10, 10)
            real_vaes_tensor =  torch.cat(real_vaes, dim=0) #(5*32, 10, 10)
            # 音频开始特殊处理
            audio_left = 0 if idx - 2 < 0 else idx - 2
            audio_right = 5 if idx - 2 < 0 else idx + 1 + 2
            audio_feats = torch.tensor(audio_feats_npy[audio_left:audio_right]).reshape(-1, 384)
            
            input_vaes_tensor = input_vaes_tensor.reshape(1, len(mode)*4, vae_hw, vae_hw).to(device)
            audio_feats = audio_feats.reshape(1, -1, 384).to(device)
            audio_feats = pe(audio_feats)
            
            return input_vaes_tensor, audio_feats
        # print(vae_ref_t1_t2.shape, hu_t0_t3.shape)
        input_vaes = []
        input_audio_feats = []
        for i in range(batchsize):
            input_vae, audio_feat = get_data(img_idx+i)
            input_vaes.append(input_vae)
            input_audio_feats.append(audio_feat)
        
        input_vaes = torch.cat(input_vaes, dim=0)
        input_audio_feats = torch.cat(input_audio_feats, dim=0)
        
        # print(input_vaes.shape, input_audio_feats.shape)
        # raise OSError()
        
        timesteps = torch.tensor([0], device=device)
        with torch.no_grad():
            vaes_pred = model(input_vaes.float(), timestep=timesteps.float(), encoder_hidden_states=input_audio_feats.float()).sample
            vaes_pred = vaes_pred.reshape(-1, 4, vae_hw, vae_hw)
            # print(vaes_pred.shape, vaes_pred[0:0,].shape)
            for i in range(batchsize):
                y = vae.decode_latents_tensor(vaes_pred[i:i+1,])
                save_image(y * 0.5 + 0.5, f'{test_dir}/{img_idx+i}.jpg')
            
                #人像放回原图
                def restore(index):
                    face = cv2.imread(f'{test_dir}/{index}.jpg')
                    full_frame = cv2.imread(f'{test_dir}/{dataName}/{index}.jpg')
                    box = np.load(os.path.join(directory, f'{index}_box.npy'))
                    affine_matrix = np.load(os.path.join(directory, f'{index}_matrix.npy'))
                    x1, y1, x2, y2 = box
                    height = int(y2 - y1)
                    width = int(x2 - x1)
                    face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    out_frame = image_processor.restorer.restore_img(full_frame, face, affine_matrix)
                    cv2.imwrite(f"{test_dir}/out{index}.jpg", out_frame)
                
                restore(img_idx+i)
                # raise OSError()
            
            print(f'\r inference out_frame => {img_idx}', end='')
            img_idx += batchsize
        
        
        
        


if __name__ == '__main__':
    test_dir = "./test"
    prefix = "out"
    inference(36000, test_dir)
    def head_mp4():
        # os.system(f"cp -rf {test_dir}/{prefix}2.jpg {test_dir}/{prefix}0.jpg")
        # os.system(f"cp -rf {test_dir}/{prefix}2.jpg {test_dir}/{prefix}1.jpg")
        os.system(f"ffmpeg -framerate 25 -i {test_dir}/{prefix}%d.jpg -c:v libx264 -pix_fmt yuv420p -y {test_dir}/outputjpg.mp4")
        os.system(f"ffmpeg -i {test_dir}/outputjpg.mp4 -i {test_dir}/output.wav -c:v copy -c:a aac -y {test_dir}/output.mp4")
        os.system(f"rm {test_dir}/*.jpg")
    head_mp4()
    
    # os.system("cp -rf ./test/out2.jpg ./test/out0.jpg")
    # os.system("cp -rf ./test/out2.jpg ./test/out0.jpg")
    # os.system(f"ffmpeg -framerate 25 -i ./test/out%d.jpg -c:v libx264 -pix_fmt yuv420p -y ./test/outputjpg.mp4")
    # os.system(f"ffmpeg -i ./test/outputjpg.mp4 -i ./test/output.wav -c:v copy -c:a aac -y ./test/output.mp4")
    # os.system("rm ./test/*.jpg")
    
    # ffmpeg -framerate 25 -i out%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output12000.mp4
    # ffmpeg -i output12000.mp4 -i output.wav -c:v copy -c:a aac -y output.mp4
    # export CUDA_VISIBLE_DEVICES="2"
    # screen bash -c "source /root/miniconda3/bin/activate facetalker && python model.py"
    # tensorboard --logdir=runs --port 9500 --bind_all
    # conda activate facetalker
    