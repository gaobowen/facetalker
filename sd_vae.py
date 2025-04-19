from diffusers import AutoencoderKL
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader




class VAE():
    """
    VAE (Variational Autoencoder) class for image processing.
    """

    def __init__(self, model_path="./models/sd-vae-ft-mse/", resized_img=384, use_float16=False, device="cuda"):
        """
        Initialize the VAE instance.

        :param model_path: Path to the trained model.
        :param resized_img: The size to which images are resized.
        :param use_float16: Whether to use float16 precision.
        """
        self.model_path = model_path
        self.vae = AutoencoderKL.from_pretrained(self.model_path)
        self.device = device
        self.vae.to(self.device)

        if use_float16:
            self.vae = self.vae.half()
            self._use_float16 = True
        else:
            self._use_float16 = False
        
        self.vae.requires_grad_(False)

        self.scaling_factor = self.vae.config.scaling_factor
        self.transform_normalize = transforms.Compose([ # -1 ~ 1
            transforms.Resize((resized_img, resized_img)),  # 调整图片大小
            transforms.ToTensor(),          # 转换为张量 0-1
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化[-1,1] Normalize = (tensor - mean) / std
        ])
        
        self.transform = transforms.Compose([ # 0~1
            transforms.Resize((resized_img, resized_img)),  # 调整图片大小
            transforms.ToTensor(),          # 转换为张量
        ])
        
        self._resized_img = resized_img
        
        
    def encode_image(self, image): # PIL Image
        input_image = self.transform_normalize(image).unsqueeze(0).to(self.device)
        # print(input_image.size())
        with torch.no_grad():
            init_latent_dist = self.vae.encode(input_image.to(self.vae.dtype)).latent_dist
            init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents
    
    def encode_normalized_image(self, input_image): # PIL Image
        # print(input_image.size())
        with torch.no_grad():
            init_latent_dist = self.vae.encode(input_image.to(self.vae.dtype)).latent_dist
            init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents
    
    def decode_latents_tensor(self, latents): # -1~1
        # 不能加 no_grad，否则无法传播梯度
        latents = (1 / self.scaling_factor) * latents
        tensor = self.vae.decode(latents.to(self.vae.dtype)).sample
        return tensor
    
def test_encode():
    img_path = "/data/split_video_25fps_imgs/TomUdall_1/0.jpg"
    image = Image.open(img_path)
    latents = vae.encode_image(image)
    print(img_path,"latents", latents.size())
    return latents
    
def test_decode():
    device="cuda:1"
    latents = torch.load("/data/gaobowen/split_video_25fps_sdvae320/1736236299779-160336/0.vae").to(device)
    # latents.to(device)
    # print(type(latents))
    path='./out.jpg'
    vae = VAE(model_path="./models/sd-vae-ft-mse/", resized_img=320, use_float16=False, device=device)
    
    tensor = vae.decode_latents_tensor(latents)
    # https://blog.csdn.net/qq_41813454/article/details/136267871
    tensor = (tensor / 2 + 0.5).clamp(0, 1) #
    
    to_pil = transforms.ToPILImage()
    
    pilimg = to_pil(tensor.detach().cpu().squeeze())
    
    pilimg.save(path)
        


class ImgDataset(Dataset):
    def __init__(self, root_dir='/data/split_video_25fps_imgs-2', resized_img=320):
        super(ImgDataset, self).__init__()
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True)
        self.num_frames = len(self.image_paths)
        self.transform = transforms.Compose([
            # transforms.Resize((resized_img, resized_img)),  # 调整图片大小
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化[-1,1] Normalize = (tensor - mean) / std   
        ])
        
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        imgpath = self.image_paths[index]
        sel_img = self.transform(Image.open(imgpath))
        return sel_img, imgpath


def preprocess_vae():
    accelerator = Accelerator()
    device = accelerator.device
    resized_img = 320
    vae = VAE(model_path="./models/sd-vae-ft-mse/", resized_img=resized_img, use_float16=False, device=device)
    
    # root_dir = "/data/gaobowen/vaildata_imgs"
    root_dir = "/data/gaobowen/split_video_25fps_sdvae320-2"
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
        if not os.path.exists(vae_path):
            latent = vae.encode_normalized_image(sel_img)
            torch.save(latent.detach().cpu(), vae_path)
        progress_bar.update(1)


if __name__ == "__main__":
    
    preprocess_vae()
    
    # test_decode()
    

    # CUDA_VISIBLE_DEVICES="3,4,5,6,7" accelerate launch sd_vae.py
    # conda activate facetalker

    
    
    

    # screen bash -c "source /root/miniconda3/bin/activate facetalker && python vae.py"
    
    # image = Image.fromarray(rgb_array, 'RGB')
    
    # crop_imgs_path = "./results/sun001_crop/"
    # latents_out_path = "./results/latents/"
    # if not os.path.exists(latents_out_path):
    #     os.mkdir(latents_out_path)

    # files = os.listdir(crop_imgs_path)
    # files.sort()
    # files = [file for file in files if file.split(".")[-1] == "png"]

    # for file in files:
    #     index = file.split(".")[0]
    #     img_path = crop_imgs_path + file
    #     latents = vae.get_latents_for_unet(img_path)
    #     print(img_path,"latents",latents.size())
    #     #torch.save(latents,os.path.join(latents_out_path,index+".pt"))
    #     #reload_tensor = torch.load('tensor.pt')
    #     #print(reload_tensor.size())
        

    