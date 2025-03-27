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


# latent 3x320x320 -> 32x10x10
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



dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()

transform = transforms.Compose([
                # transforms.Resize((resized_img, resized_img)),  # 调整图片大小
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化[-1,1] Normalize = (tensor - mean) / std   
            ])


def inference(step_num):
    
    ckpt = f"/data/gaobowen/facetalker/output/checkpoint-{step_num}/model.safetensors"
    
    # unet.load_state_dict(torch.load(ckpt, map_location=device))
    model = SanaTransformer2DModel(**config)
    model.load_state_dict(load_file(ckpt))
    model = model.to(device).eval()
    # AlexandriaOcasioCortez_1 AustinScott2_1 BarackObama_1 1733208337445-160927 1736236299779-160336
    dataName = "1736236299779-160336" # 
    
    img_dir = f'/data/gaobowen/split_video_25fps_imgs-2/{dataName}'
    image_paths = glob.glob(os.path.join(img_dir, '*.jpg'), recursive=True)
    
    mp4_path = f"/data/gaobowen/split_video_25fps/{dataName}.mp4"
    
    # filename = os.path.basename(imgpath)
    # img_idx = int(filename.replace('.jpg', ''))
    
    directory = img_dir
    
    whisper_npy = np.load('./test/output_whisper.npy')
    max_count, seqlen, dim = whisper_npy.shape
    print(whisper_npy.shape)
    
    
    # raise OSError()
    
        
    # R       I2 I3 I4 I5 I6
    #   A0 A1 A2 A3 A4 A5 A6 A7 A8
    max_img_idx = max_count - 7
    
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
        mode = "Rxxxxx"
        for char in mode:
            if char == 'R':
                ref_vae = torch.load(os.path.join(directory, f'{ref_idx}.vae')).reshape(-1, 10, 10)
                input_vaes.append(ref_vae)
            elif char == 'I':
                sel_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}.vae')).reshape(-1, 10, 10)
                input_vaes.append(sel_vae)
                real_vaes.append(sel_vae)
                seq_idx += 1
            elif char == 'x':
                sel_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}_mask.vae')).reshape(-1, 10, 10)
                input_vaes.append(sel_vae)
                real_vaes.append(torch.load(os.path.join(directory, f'{img_idx + seq_idx}.vae')).reshape(-1, 10, 10))
                seq_idx += 1
            # print(img_idx + seq_idx)
        input_vaes_tensor =  torch.cat(input_vaes, dim=0) # (6*32, 10, 10)
        real_vaes_tensor =  torch.cat(real_vaes, dim=0) #(5*32, 10, 10)
        audio_feats = torch.tensor(whisper_npy[img_idx-2:img_idx + 7]).reshape(-1, 384) # (90 384) 
        
        input_vaes_tensor = input_vaes_tensor.reshape(1, 6*32, 10, 10).to(device)
        audio_feats = audio_feats.reshape(1, 90, 384).to(device)
        timesteps = torch.ones(1).to(device) * 0.31831
        # print(vae_ref_t1_t2.shape, hu_t0_t3.shape)
        
        # raise OSError()
        
        with torch.no_grad():
            # B 32*5 10 10
            vaes_pred = model(hidden_states=input_vaes_tensor.float(), timestep=timesteps.float(), encoder_hidden_states=audio_feats.float()).sample
            vaes_pred = vaes_pred.half()
            for i in range(5):
                y = dc_ae.decode(vaes_pred[:,i*32:(i+1)*32], return_dict=False)[0]
                save_image(y * 0.5 + 0.5, f'./test/{img_idx+i}.jpg')
            
            # # 人像放回原图，左上右下
            # def copyhead(idx):
            #     box = np.load(os.path.join(directory, f'{idx}_box.npy'))
            #     # print(box)
            #     x = box[0]
            #     y = box[1]
            #     w = box[2]
            #     h = box[3]
            #     # H W C
            #     head1 = cv2.imread(f'./test/{idx}.jpg')
            #     head1 = cv2.resize(head1, (w, h), interpolation=cv2.INTER_LANCZOS4)
            #     # print(head1.shape)
            #     img1 = cv2.imread(f'./test/full{idx}.jpg')
            #     # print(img1.shape)
            #     # print(img1[y:y+h,x:x+w,:].shape)
            #     img1[y:y+h,x:x+w,:] = head1
                
            #     cv2.imwrite(f'./test/out{idx}.jpg', img1)
            
            # copyhead(img_idx)
            # copyhead(img_idx+1)
            
            print(f'inference => {img_idx}')
        
        img_idx += 5


if __name__ == '__main__':
    inference(58000)
    def head_mp4():
        os.system("cp -rf ./test/2.jpg ./test/0.jpg")
        os.system("cp -rf ./test/2.jpg ./test/1.jpg")
        os.system(f"ffmpeg -framerate 25 -i ./test/%d.jpg -c:v libx264 -pix_fmt yuv420p -y ./test/outputjpg.mp4")
        os.system(f"ffmpeg -i ./test/outputjpg.mp4 -i ./test/output.wav -c:v copy -c:a aac -y ./test/output.mp4")
        os.system("rm ./test/*.jpg")
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
    