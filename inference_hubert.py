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
from safetensors.torch import load_file

import os
import argparse
import itertools
import math
import random
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# latent 384x384 -> 32x12x12
num_frames = 10
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
    "caption_channels": 1024,
}



dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()

transform = transforms.Compose([
                # transforms.Resize((resized_img, resized_img)),  # 调整图片大小
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化[-1,1] Normalize = (tensor - mean) / std   
            ])



def inference(step_num, test_dir):
    
    ckpt = f"/data/gaobowen/facetalker/output_hubert/checkpoint-{step_num}/model.safetensors"
    
    # unet.load_state_dict(torch.load(ckpt, map_location=device))
    model = SanaTransformer2DModel(**config)
    model.load_state_dict(load_file(ckpt))
    model = model.to(device).eval()
    # AlexandriaOcasioCortez_1 AustinScott2_1 BarackObama_1 1733208337445-160927 1736236299779-160336
    dataName = "1736236299779-160336" # 
    
    img_dir = f'/data/gaobowen/split_video_25fps_stable/{dataName}'
    image_paths = glob.glob(os.path.join(img_dir, '*.jpg'), recursive=True)
    
    mp4_path = f"/data/gaobowen/split_video_25fps/{dataName}.mp4"
    
    # filename = os.path.basename(imgpath)
    # img_idx = int(filename.replace('.jpg', ''))
    
    directory = img_dir
    
    audio_feats_npy = np.load(f"{test_dir}/output_hu.npy")
    max_count, dim = audio_feats_npy.shape
    print(audio_feats_npy.shape)
    vae_hw = 12
    
    # raise OSError()
    mode = "Rxxxxxxxxxx"
    # R  I2 I3 I4 I5 I6
    #    A2 A3 A4 A5 A6
    max_img_idx = max_count - len(mode) - 1
    img_seq_len = len(mode) - 1
    
    # vide_capture = cv2.VideoCapture(mp4_path)
    # index = 0
    # while index <= max_img_idx + 1:
    #     ret, image = vide_capture.read()
    #     if ret:
    #         cv2.imwrite(f'./test/full{index}.jpg', image)
    #         index += 1
    

    img_idx = 2
    while img_idx < max_img_idx:
        ref_idx = img_idx - 2
        
        input_vaes = []
        real_vaes = []
        real_imgs = []
        sel_img = None
        seq_idx = 0
        # file_mode = "vae" # jpg vae
        mode = "Rxxxxxxxxxx"
        for char in mode:
            if char == 'R':
                ref_vae = torch.load(os.path.join(directory, f'{ref_idx}.vae')).reshape(-1, vae_hw, vae_hw)
                input_vaes.append(ref_vae)
            elif char == 'I':
                sel_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}.vae')).reshape(-1, vae_hw, vae_hw)
                input_vaes.append(sel_vae)
                real_vaes.append(sel_vae)
                seq_idx += 1
            elif char == 'x':
                sel_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}_mask.vae')).reshape(-1, vae_hw, vae_hw)
                input_vaes.append(sel_vae)
                real_vaes.append(torch.load(os.path.join(directory, f'{img_idx + seq_idx}.vae')).reshape(-1, vae_hw, vae_hw))
                seq_idx += 1
            # print(img_idx + seq_idx)
        input_vaes_tensor =  torch.cat(input_vaes, dim=0) # (6*32, 10, 10)
        real_vaes_tensor =  torch.cat(real_vaes, dim=0) #(5*32, 10, 10)
        audio_feats = torch.tensor(audio_feats_npy[img_idx:img_idx + img_seq_len]).reshape(-1, 1024)
        
        input_vaes_tensor = input_vaes_tensor.reshape(1, len(mode)*32, vae_hw, vae_hw).to(device)
        audio_feats = audio_feats.reshape(1, -1, 1024).to(device)
        timesteps = torch.ones(1).to(device) * 0.31831
        # print(vae_ref_t1_t2.shape, hu_t0_t3.shape)
        
        # raise OSError()
        
        with torch.no_grad():
            # B 32*5 10 10
            vaes_pred = model(hidden_states=input_vaes_tensor.float(), timestep=timesteps.float(), encoder_hidden_states=audio_feats.float()).sample
            vaes_pred = vaes_pred.half().reshape(-1, 32, vae_hw, vae_hw)
            for i in range(img_seq_len):
                y = dc_ae.decode(vaes_pred[i:(i+1)], return_dict=False)[0]
                save_image(y * 0.5 + 0.5, f'{test_dir}/{img_idx+i}.jpg')
            
            # # 人像放回原图，左上右下
            # def copyhead(idx):
            #     box = np.load(os.path.join(directory, f'{idx}_box.npy'))
            
            # copyhead(img_idx)
            # copyhead(img_idx+1)
            
            print(f'inference => {img_idx}')
        
        img_idx += img_seq_len


if __name__ == '__main__':
    test_dir = "./test_hubert"
    inference(10000, test_dir)
    def head_mp4():
        os.system(f"cp -rf {test_dir}/2.jpg {test_dir}/0.jpg")
        os.system(f"cp -rf {test_dir}/2.jpg {test_dir}/1.jpg")
        os.system(f"ffmpeg -framerate 25 -i {test_dir}/%d.jpg -c:v libx264 -pix_fmt yuv420p -y {test_dir}/outputjpg.mp4")
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
    
    # screen bash -c "source /root/miniconda3/bin/activate facetalker && python model.py"
    # tensorboard --logdir=runs --port 9500 --bind_all
    # conda activate facetalker
    