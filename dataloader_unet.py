import os
import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import random
import glob
import torchvision.transforms as transforms
import json
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir='/data/gaobowen/split_video_25fps_stable', mode='xR', only_vae = True, audio_feats_type="whisper", audio_window_add = 2, imgHW = 320):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*_mask.jpg'), recursive=True)
        self.num_frames = len(self.image_paths)
        # 分为不同的难度
        # RIIIII RIIxII RxIxIx Rxxxxx
        self.mode = mode
        self.only_vae = only_vae
        self.audio_feats_type = audio_feats_type
        self.audio_window_add = audio_window_add
        self.imgHW = imgHW
        self.transform = transforms.Compose([
            # transforms.Resize((resized_img, resized_img)),  # 调整图片大小
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化[-1,1] Normalize = (tensor - mean) / std   
        ])
        
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        imgpath = self.image_paths[index]
        imgpath = imgpath.replace('_mask.jpg', '.jpg')
        filename = os.path.basename(imgpath)
        directory = os.path.dirname(imgpath)
        
        # print(directory)
        img_idx = int(filename.replace('.jpg', ''))

        if self.audio_feats_type == "whisper":
            audio_feats_npy = np.load(os.path.join(directory, 'output_whisper.npy'))
            max_count, seqlen, dim = audio_feats_npy.shape
        elif self.audio_feats_type == "hubert":
            audio_feats_npy = np.load(os.path.join(directory, 'output_hu.npy'))
            max_count, dim = audio_feats_npy.shape

        img_seq_len = len(self.mode) - 1
        #          I0 I1 I2 I3 R
        #  ... A A A0 A1 A2 A3 A A ...
        max_img_idx = max_count - len(self.mode) - self.audio_window_add
        
        if img_idx <= self.audio_window_add or img_idx > max_img_idx:
            img_idx = random.randint(self.audio_window_add, max_img_idx)
        def check_exists():
            for idx in range(img_idx, img_idx + img_seq_len):
                if not os.path.exists(os.path.join(directory, f'{idx}.jpg')):
                    return False
            return True
        
        while not check_exists():
            img_idx = random.randint(self.audio_window_add, max_img_idx)
            
        ref_idx = random.randint(0, max_img_idx)
        while (abs(ref_idx - img_idx) < 50 or
               not os.path.exists(os.path.join(directory, f'{ref_idx}.jpg'))):
            ref_idx = random.randint(0, max_img_idx)
        
        
        input_vaes = []
        real_vaes = []
        real_imgs = []
        sel_img = None
        seq_idx = 0
        # file_mode = "vae" # jpg vae
        if self.audio_feats_type == "whisper":
            vae_hw = int(self.imgHW / 8)
        elif self.audio_feats_type == "hubert":
            vae_hw = int(self.imgHW / 32)
        
        for char in self.mode:
            if char == 'R':
                # input_imgs.append(self.transform(Image.open(os.path.join(directory, f'{ref_idx}.jpg'))))
                ref_vae = torch.load(os.path.join(directory, f'{ref_idx}.vae')).reshape(-1, vae_hw, vae_hw)
                input_vaes.append(ref_vae)
            elif char == 'I':
                sel_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}.vae')).reshape(-1, vae_hw, vae_hw)
                if not self.only_vae:
                    real_img = self.transform(Image.open(os.path.join(directory, f'{img_idx + seq_idx}.jpg')))
                    real_imgs.append(real_img)
                input_vaes.append(sel_vae)
                real_vaes.append(sel_vae)
                seq_idx += 1
            elif char == 'x':
                sel_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}_mask.vae')).reshape(-1, vae_hw, vae_hw)
                if not self.only_vae:
                    real_img = self.transform(Image.open(os.path.join(directory, f'{img_idx + seq_idx}.jpg')))
                    real_imgs.append(real_img)
                input_vaes.append(sel_vae)
                # real_vae = torch.load(os.path.join(directory, f'{img_idx + seq_idx}.vae')).reshape(-1, vae_hw, vae_hw)
                # real_vaes.append(real_vae)
                seq_idx += 1
            # print(img_idx + seq_idx)
        input_vaes_tensor =  torch.cat(input_vaes, dim=0) # (n+1*32, 12, 12)
        # real_vaes_tensor =  torch.cat(real_vaes, dim=0) # (n*32, 12, 12)
        if self.audio_feats_type == "whisper":
            audio_feats = torch.tensor(audio_feats_npy[img_idx - self.audio_window_add:img_idx + img_seq_len + self.audio_window_add]).reshape(-1, 384) # (50 384) 
        elif self.audio_feats_type == "hubert":
            audio_feats = torch.tensor(audio_feats_npy[img_idx - self.audio_window_add:img_idx + img_seq_len + self.audio_window_add]).reshape(-1, 1024)
        
        real_imgs_tensor = "None"
        if not self.only_vae:
            real_imgs_tensor = torch.cat(real_imgs, dim=0) #(5*3, 320, 320)
        
        
            
        return input_vaes_tensor, audio_feats, real_imgs_tensor
        
        

if __name__ == "__main__":
    from torchvision.utils import save_image
    from diffusers import AutoencoderDC
    
    device = torch.device("cuda")

    # dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/")
    # dc_ae = dc_ae.to(device).half().eval()
    
    dataset = MyDataset(root_dir='/data/gaobowen/split_video_25fps_sdvae320', mode='xR', only_vae = False, 
        audio_feats_type="whisper", audio_window_add = 2, imgHW = 320)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
    
                                    
    for step, (input_vaes_tensor, audio_feats, real_imgs_tensor) in enumerate(dataloader):
        print(input_vaes_tensor.shape, audio_feats.shape, real_imgs_tensor.shape)
        # target_vaes_tensor = real_vaes_tensor
        # target_vaes_tensor = target_vaes_tensor.half().to(device) #(batch, 5*32, 10, 10)
        # target_vaes_tensor = target_vaes_tensor.reshape(-1, 32, 10, 10) #(batch * 5, 32, 10, 10)
        # print(target_vaes_tensor.shape)
        # # latent = dc_ae.encode(input_imgs_pt, return_dict=False)[0]
        # # print(latent.shape)
        # y = dc_ae.decode(target_vaes_tensor, return_dict=False)[0]
        # print(y.shape) #(batch*5, 3, 320, 320)
        # for i in range(y.shape[0]):
        #     save_image(y[i] * 0.5 + 0.5, f"./test/{i}.jpg")
        
        
        # real_imgs_tensor  = real_imgs_tensor.reshape(-1, 3, 320, 320)
        # print(real_imgs_tensor.shape)
        # for i in range(real_imgs_tensor.shape[0]):
        #     save_image(real_imgs_tensor[i] * 0.5 + 0.5, f"./test/{i}real.jpg")
        break
        