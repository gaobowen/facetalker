# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from torchvision import transforms
import cv2
from einops import rearrange
import mediapipe as mp
import torch
import numpy as np
from typing import Union
from affine_transform import AlignRestore, laplacianSmooth
import face_alignment

from pathlib import Path
import glob
from tqdm import tqdm



from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def load_fixed_mask(resolution: int, mask_image_path="") -> torch.Tensor:
    mask_image_path = f"{os.path.dirname(os.path.abspath(__file__))}/mask.png"
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, mask: str = "fix_mask", device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.mask = mask

        if mask in ["mouth", "face", "eye"]:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
        if mask == "fix_mask":
            self.face_mesh = None
            self.smoother = laplacianSmooth()
            self.restorer = AlignRestore()

            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image

            if device != "cpu":
                self.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
                )
                self.face_mesh = None
            else:
                # self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
                self.face_mesh = None
                self.fa = None

    def detect_facial_landmarks(self, image: np.ndarray):
        height, width, _ = image.shape
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            raise RuntimeError("Face not detected")
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image
        landmark_coordinates = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark
        ]  # x means width, y means height
        return landmark_coordinates

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask == "mouth" or self.mask == "face":
            landmark_coordinates = self.detect_facial_landmarks(image)
            if self.mask == "mouth":
                surround_landmarks = mouth_surround_landmarks
            else:
                surround_landmarks = face_surround_landmarks

            points = [landmark_coordinates[landmark] for landmark in surround_landmarks]
            points = np.array(points)
            mask = np.ones((self.resolution, self.resolution))
            mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
        elif self.mask == "half":
            mask = torch.ones((self.resolution, self.resolution))
            height = mask.shape[0]
            mask[height // 2 :, :] = 0
            mask = mask.unsqueeze(0)
        elif self.mask == "eye":
            mask = torch.ones((self.resolution, self.resolution))
            landmark_coordinates = self.detect_facial_landmarks(image)
            y = landmark_coordinates[195][1]
            mask[y:, :] = 0
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Invalid mask type")

        image = image.to(dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * mask
        mask = 1 - mask

        return pixel_values, masked_pixel_values, mask

    def affine_transform(self, image: torch.Tensor, allow_multi_faces: bool = True) -> np.ndarray:
        # image = rearrange(image, "c h w-> h w c").numpy()
        if self.fa is None:
            landmark_coordinates = np.array(self.detect_facial_landmarks(image))
            lm68 = mediapipe_lm478_to_face_alignment_lm68(landmark_coordinates)
        else:
            detected_faces = self.fa.get_landmarks(image)
            if detected_faces is None:
                raise RuntimeError("Face not detected")
            if not allow_multi_faces and len(detected_faces) > 1:
                raise RuntimeError("More than one face detected")
            lm68 = detected_faces[0]

        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)
        # print(lmk3_)
        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
        )
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        # face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        if self.mask == "fix_mask":
            results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]
        else:
            results = [self.preprocess_one_masked_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """
    lm478: [B, 478, 3] or [478,3]
    """
    # lm478[..., 0] *= W
    # lm478[..., 1] *= H
    landmarks_extracted = []
    for index in landmark_points_68:
        x = lm478[index][0]
        y = lm478[index][1]
        landmarks_extracted.append((x, y))
    return np.array(landmarks_extracted)


landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


# Refer to https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
mouth_surround_landmarks = [
    164,
    165,
    167,
    92,
    186,
    57,
    43,
    106,
    182,
    83,
    18,
    313,
    406,
    335,
    273,
    287,
    410,
    322,
    391,
    393,
]

face_surround_landmarks = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    435,
    433,
    411,
    425,
    423,
    327,
    326,
    94,
    97,
    98,
    203,
    205,
    187,
    213,
    215,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
]



class VideoDataset(Dataset):
    def __init__(self, root_dir='/data/gaobowen/split_video_25fps'):
        super(VideoDataset, self).__init__()
        self.root_dir = root_dir
        self.video_paths = glob.glob(os.path.join(root_dir, '**/*.mp4'), recursive=True)
        self.num = len(self.video_paths)
        
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        vidpath = self.video_paths[index]
        
        return vidpath

# os.environ['CUDA_VISIBLE_DEVICES']="5"
if __name__ == "__main__":
    from torchvision.utils import save_image

    accelerator = Accelerator()
    device = accelerator.device

    print(str(device))

    # raise OSError()

    resolution = 320
    # data_dir = "/data/gaobowen/split_video_25fps"
    # out_dir = "/data/gaobowen/split_video_25fps_sdvae320"
    data_dir = "/data/gaobowen/vaildata"
    out_dir = "/data/gaobowen/vaildata_imgs"
    dataset = VideoDataset(root_dir=data_dir)
    total = dataset.num
    print(data_dir, total)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    dataloader = accelerator.prepare(dataloader)

    # 只对主进程生效 disable=not accelerator.is_local_main_process
    progress_bar = tqdm(range(0, len(dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    image_processor = ImageProcessor(resolution, mask="fix_mask", device=str(device), )
    '''
    /data/split_video_25fps/BarackObama_1.mp4
    /data/laihuadata/1742353006021-112652.mp4
    '''
    def save_video_face(video_path, save_dir):
        audio_path = os.path.join(save_dir, 'output.wav')
        if os.path.exists(audio_path): return
        
        video = cv2.VideoCapture(video_path)
        mask_image = cv2.imread("./mask.png")
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
        index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # cv2.imwrite("image.jpg", frame)
            # image_processor.fa.get_landmarks(image)
            # frame = rearrange(torch.Tensor(frame).type(torch.uint8), "h w c ->  c h w")
            # face, masked_face, _ = image_processor.preprocess_fixed_mask_image(frame, affine_transform=True)
            # 输入 numpy格式h w c，输出 torch格式c h w
            try:
                face, box, affine_matrix = image_processor.affine_transform(frame)
            except:
                index += 1
                continue
                
            # print(type(face), type(box), type(affine_matrix))
            cv2.imwrite(os.path.join(save_dir, f"{index}.jpg"), face)
            npbox = np.array(box)
            np.save(os.path.join(save_dir, f'{index}_box.npy'), npbox)
            np.save(os.path.join(save_dir, f'{index}_matrix.npy'), affine_matrix)
            face = face * mask_image
            cv2.imwrite(os.path.join(save_dir, f"{index}_mask.jpg"), face)
            
            def restore():
                box = np.load(os.path.join(save_dir, f'{index}_box.npy'))
                affine_matrix = np.load(os.path.join(save_dir, f'{index}_matrix.npy'))
                x1, y1, x2, y2 = box
                height = int(y2 - y1)
                width = int(x2 - x1)
                face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
                out_frame = image_processor.restorer.restore_img(frame, face, affine_matrix)
                cv2.imwrite(f"../test/out{index}.jpg", out_frame)

            index += 1
            # print(f"\r{index}", end="")
            # break
        
        os.system(f"ffmpeg -loglevel error -i {video_path} -ac 1 -ar 16000 -vn -y {audio_path}")

    
    for step, (path) in enumerate(dataloader):
        filepath = path[0]
        print(filepath)
        save_dir = os.path.join(out_dir, Path(filepath).stem)
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)
        save_video_face(filepath, save_dir)
        progress_bar.update(1)


    # export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" && accelerate launch image_processor.py
    # export CUDA_VISIBLE_DEVICES="1" && accelerate launch image_processor.py





    # mp4_dir = "/data/gaobowen/split_video_25fps"
    # out_dir = "/data/gaobowen/split_video_25fps_stable"
    # id_mp4s = glob.glob(f"{mp4_dir}/*.mp4", recursive=True)
    
    # print(mp4_dir, out_dir, len(id_mp4s))

    # def run_all_data(id_mp4s):
    #     for filepath in tqdm(id_mp4s):
    #         save_dir = os.path.join(out_dir, Path(filepath).stem)
    #         os.makedirs(save_dir, exist_ok=True)
    #         save_video_face(filepath, save_dir)
    #         # break

    # run_all_data(id_mp4s)
    
    # face = (rearrange(face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    # cv2.imwrite("./face.jpg", face)
    # masked_face = (rearrange(masked_face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    # cv2.imwrite("masked_face.jpg", masked_face)
    # ffmpeg -framerate 25 -i ../test/out%d.jpg -c:v libx264 -y ../outputjpg.mp4
    


    # export CUDA_VISIBLE_DEVICES="7"
