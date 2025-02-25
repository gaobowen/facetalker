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
    def __init__(self, root_dir='/data/split_video_25fps_imgs-2', mode='Rxxxxx'):
        """
        Args:
            root_dir (string): The directory path to the dataset images.
        """
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*_mask.jpg'), recursive=True)
        self.num_frames = len(self.image_paths)
        # 分为不同的难度训练
        # RIIIII RIIxII RxIxIx Rxxxxx
        self.mode = mode
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
        print(directory)
        img_idx = int(filename.replace('.jpg', ''))

        whisper_npy = np.load(os.path.join(directory, 'output_whisper.npy'))
        max_count, seqlen, dim = whisper_npy.shape
        
        # R       I2 I3 I4 I5 I6
        #   A0 A1 A2 A3 A4 A5 A6 A7 A8
        max_img_idx = max_count - 7
        
        if img_idx < 2 or img_idx > max_img_idx:
            img_idx = random.randint(2, max_img_idx)
        def check_exists():
            for idx in range(img_idx, img_idx + 5):
                if not os.path.exists(os.path.join(directory, f'{idx}.jpg')):
                    return False
            return True
        
        while not check_exists():
            img_idx = random.randint(2, max_img_idx)
            
        ref_idx = random.randint(0, max_img_idx)
        while (abs(ref_idx - img_idx) < 15 or
               not os.path.exists(os.path.join(directory, f'{ref_idx}.jpg'))):
            ref_idx = random.randint(0, max_img_idx)
        
        
        input_imgs = []
        real_imgs = []
        seq_idx = 0
        for char in self.mode:
            if char == 'R':
                input_imgs.append(self.transform(Image.open(os.path.join(directory, f'{ref_idx}.jpg'))))
            elif char == 'I':
                sel_img = self.transform(Image.open(os.path.join(directory, f'{img_idx + seq_idx}.jpg')))
                input_imgs.append(sel_img)
                real_imgs.append(sel_img)
                seq_idx += 1
            elif char == 'x':
                imgs.append(self.transform(Image.open(os.path.join(directory, f'{img_idx + seq_idx}_mask.jpg'))))
                real_imgs.append(self.transform(Image.open(os.path.join(directory, f'{img_idx + seq_idx}.jpg'))))
                seq_idx += 1
            print(img_idx + seq_idx)
        input_imgs_pt = torch.stack(input_imgs, dim=0) # (6 3 H W)
        real_imgs_pt = torch.stack(real_imgs, dim=0)
        
        audio_feats = torch.tensor(whisper_npy[img_idx-2:img_idx + 7]).reshape(-1, 384) # (90 384) 
        
        return input_imgs_pt, audio_feats, real_imgs_pt
        
        

if __name__ == "__main__":
    from torchvision.utils import save_image
    from diffusers import AutoencoderDC
    
    device = torch.device("cuda")

    dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/")
    dc_ae = dc_ae.to(device).half().eval()
    
    dataset = MyDataset(root_dir='/data/split_video_25fps_imgs-2', mode='RIIIII')

    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
                                    
    for batch in dataloader:
        input_imgs_pt, audio_feats, real_imgs_pt = batch
        input_imgs_pt = input_imgs_pt.half().to(device) #[1, 6, 3, 384, 384]
        input_imgs_pt = torch.squeeze(input_imgs_pt, dim=0)
        print(input_imgs_pt.shape)
        latent = dc_ae.encode(input_imgs_pt, return_dict=False)[0]
        print(latent.shape)
        y = dc_ae.decode(latent, return_dict=False)[0]
        print(y.shape)
        for i in range(y.shape[0]):
            save_image(y[i] * 0.5 + 0.5, f"./test/{i}.jpg")
        break
        