
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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class ImgDataset(Dataset):
    def __init__(self, root_dir='/data/split_video_25fps_imgs-2', resized_img=384):
        super(ImgDataset, self).__init__()
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True)
        self.num_frames = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((resized_img, resized_img)),  # 调整图片大小
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化[-1,1] Normalize = (tensor - mean) / std   
        ])
        
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        imgpath = self.image_paths[index]
        sel_img = self.transform(Image.open(imgpath)).half()
        return sel_img, imgpath

def vae_preprocess(root_dir, resized_img):
    accelerator = Accelerator()
    
    device = accelerator.device

    dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()
    dc_ae.requires_grad_(False)

    dataset = ImgDataset(root_dir=root_dir, resized_img=resized_img)
    total = dataset.num_frames
    print(root_dir, total)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    dataloader = accelerator.prepare(dataloader)

    # 只对主进程生效 disable=not accelerator.is_local_main_process
    progress_bar = tqdm(range(0, len(dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for step, (sel_img, path) in enumerate(dataloader):
        # path 是 tuple，长度为 batch_size
        imgpath = path[0]
        vae_path = imgpath.replace(".jpg", ".vae")
        if not os.path.exists(vae_path) or True:
            latent = dc_ae.encode(sel_img, return_dict=False)[0].half()
            torch.save(latent.detach().cpu(), vae_path)
        progress_bar.update(1)



def vae_preprocess_copy(data_dir, target_dir, resized_img):
    accelerator = Accelerator()
    
    device = accelerator.device

    dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()
    dc_ae.requires_grad_(False)

    dataset = ImgDataset(root_dir=data_dir, resized_img=resized_img)
    total = dataset.num_frames
    print(data_dir, total)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    dataloader = accelerator.prepare(dataloader)

    # 只对主进程生效 disable=not accelerator.is_local_main_process
    progress_bar = tqdm(range(0, len(dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for step, (sel_img, path) in enumerate(dataloader):
        # path 是 tuple，长度为 batch_size
        imgpath = path[0]
        target_imgpath = imgpath.replace(data_dir, target_dir)
        os.makedirs(os.path.dirname(target_imgpath), exist_ok=True)
        vae_path = target_imgpath.replace(".jpg", ".vae")
        # print(target_imgpath, imgpath, sel_img.shape)
        # raise OSError()
        if not os.path.exists(vae_path) or True:
            latent = dc_ae.encode(sel_img, return_dict=False)[0].half()
            torch.save(latent.detach().cpu(), vae_path)
            save_image(sel_img * 0.5 + 0.5, target_imgpath)
        progress_bar.update(1)


def test():
    image = Image.open("./out.jpg")
    x = transform_normalize(image)[None].to(device)
    latent = dc_ae.encode(x, return_dict=False)[0]
    print(latent.shape)
    y = dc_ae.decode(latent, return_dict=False)[0]
    save_image(y * 0.5 + 0.5, "demo_dc_ae.jpg")
    
def check_preprocess():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_normalize = transforms.Compose([ # -1 ~ 1
            #transforms.Resize((resizeHW, resizeHW)),  # 调整图片大小
            transforms.ToTensor(),          # 转换为张量 0-1
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化[-1,1] Normalize = (tensor - mean) / std
        ])
    dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(device).eval()
    # raise OSError("AA")
    idx_jpgs = glob.glob(f"/data/gaobowen/split_video_25fps_imgs-2/*/*.jpg", recursive=True)
    for jpgpath in tqdm(idx_jpgs):
        vae_path = jpgpath.replace(".jpg", ".vae")
        if os.path.exists(vae_path): continue
        x = transform_normalize(Image.open(jpgpath))[None].to(device)
        latent = dc_ae.encode(x, return_dict=False)[0]
        torch.save(latent.detach().cpu(), vae_path)

if __name__ == "__main__":
    # mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers

    
    # vae_preprocess("/data/gaobowen/split_video_25fps_stable_320", 320)
    
    # vae_preprocess_copy("/data/gaobowen/split_video_25fps_stable", "/data/gaobowen/split_video_25fps_stable_320", 320)
    
    
    # audiopaths = glob.glob(os.path.join("/data/gaobowen/split_video_25fps_stable", '**/output.wav'), recursive=True)
    # for path in tqdm(audiopaths):
    #     savepath = path.replace("/data/gaobowen/split_video_25fps_stable", "/data/gaobowen/split_video_25fps_stable_320")
    #     os.system(f"cp -rf {path} {savepath}")
    # check_preprocess()
    
    # export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" && accelerate launch vae.py
    
    # print('ok')
    
    # PyTorch scaled dot-product attention (SDPA). 
    # This attention implementation is activated by default for PyTorch versions 2.1.1 or greater.