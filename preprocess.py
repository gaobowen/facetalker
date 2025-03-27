import glob
import os
import cv2
import time
import numpy as np
import os
import argparse
from tqdm import tqdm
import shutil

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pickle
from PIL import Image

# from Peppa_Pig_Face_Landmark.Skps import FaceAna

# facer = FaceAna()

base_options = python.BaseOptions(model_asset_path='./models/mediapipe/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=False,
                                    output_facial_transformation_matrixes=False,
                                    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

img_resize = 320

def read_video(video_path, save_dir):
    audio_path = os.path.join(save_dir, 'output.wav')
    # wav to hubert
    
    if os.path.exists(audio_path): return
    
    vide_capture=cv2.VideoCapture(video_path)
    index = 0
    window_x = None
    window_y = None
    window_w = None
    window_h = None
    
    while 1:
        ret, image = vide_capture.read()
        if ret:
            # try:
                # print('image.shape', image.shape) # H W C
                H, W, C = image.shape
                rgb_frame = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                detection_result = detector.detect(rgb_frame)
                if len(detection_result.face_landmarks) < 1:
                    index += 1
                    continue
                allkeys = detection_result.face_landmarks[0]

                # 1鼻子，左侧脸234 右侧脸454 眉心8 下巴152
                nose = [int(allkeys[1].x * W), int(allkeys[1].y * H)]
                left = [int(allkeys[234].x * W), int(allkeys[234].y * H)]
                right = [int(allkeys[454].x * W), int(allkeys[454].y * H)]
                top = [int(allkeys[8].x * W), int(allkeys[8].y * H)]
                bottom = [int(allkeys[152].x * W), int(allkeys[152].y * H)]
                
                keypts = [nose, left, right, top, bottom]
                # cv2.circle(image, (nose[0], nose[1]), 3, (0, 255, 0), -1)
                # cv2.circle(image, (left[0], left[1]), 3, (0, 255, 0), -1)
                # cv2.circle(image, (right[0], right[1]), 3, (0, 255, 0), -1)
                # cv2.circle(image, (top[0], top[1]), 3, (0, 255, 0), -1)
                # cv2.circle(image, (bottom[0], bottom[1]), 3, (0, 255, 0), -1)
                
                nparr = np.array(keypts)
                npmin = np.min(nparr, axis=0)
                npmax = np.max(nparr, axis=0)
                # 拓宽一点边界 截取人脸
                box_l = max(npmin[0]-16, 0)
                box_t = max(npmin[1]-1, 0)
                box_r = min(npmax[0]+16, W)
                box_b = min(npmax[1]+int((bottom[1]-nose[1])/3), H)
                
                x = box_l
                y = box_t
                w = box_r - box_l
                h = box_b - box_t
                
                # 平滑窗口，减少截图抖动
                if window_x is None:
                    window_x = [x, x, x, x, x, x, x, x, x, x]
                    window_y = [y, y, y, y, y, y, y, y, y, y]
                    window_w = [w, w, w, w, w, w, w, w, w, w]
                    window_h = [h, h, h, h, h, h, h, h, h, h]
                
                def average(val, window):
                    window.pop(0)
                    window.append(val)
                    return int(sum(window) / len(window))
                
                x = average(x, window_x)
                y = average(y, window_y)
                w = average(w, window_w)
                h = average(h, window_h)
                
                crop = image[y:y+h+1, x:x+w+1, :]

                crop = cv2.resize(crop, (img_resize, img_resize), interpolation=cv2.INTER_LANCZOS4)
                
                cv2.imwrite(os.path.join(save_dir, f'{index}.jpg'), crop)
                
                # 嘴部mask
                offset = np.array([[x, y]])
                relative_keys5 = nparr - offset
                mask = np.ones((h, w, 1), dtype=np.uint8)
                n = relative_keys5[0]
                l = relative_keys5[1]
                r = relative_keys5[2]
                m = relative_keys5[3]
                b = relative_keys5[4]
                bh = int((b[1]+ h)/2)
                pts = np.array([l, n, r, (r[0], bh), (l[0], bh)], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 0)
                
                # mask = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_LANCZOS4)
                # mask = mask.reshape((1, 1, 32, 32))
                # np.save(os.path.join(save_dir, f'{index}_mask.npy'), mask)
                
                mask = cv2.resize(mask, (img_resize, img_resize), interpolation=cv2.INTER_LANCZOS4)
                mask = mask.reshape((img_resize, img_resize, 1))
                crop = crop * mask
                cv2.imwrite(os.path.join(save_dir, f'{index}_mask.jpg'), crop)
                
                
                # 基于图片的坐标
                np.save(os.path.join(save_dir, f'{index}_lms.npy'), keypts)
                
                full_box = np.array([x, y, w, h])
                np.save(os.path.join(save_dir, f'{index}_box.npy'), full_box)
                
                index += 1

        else:
            print(f"end {index} \r")
            break
    os.system(f"ffmpeg -loglevel error -i {video_path} -ac 1 -ar 16000 -vn {audio_path}")
    

prefix = "/data/gaobowen/split_video_25fps"
outdir = "/data/gaobowen/split_video_25fps_imgs-2"

id_mp4s = glob.glob(f"{prefix}/*.mp4", recursive=True)
# id_mp4s = [f'{prefix}/BarackObama_1.mp4']

print(len(id_mp4s))

if __name__ == "__main__":
    # 卡尔曼滤波 稳定人脸数据集
    # https://blog.51cto.com/u_16213379/12991281
    def run_all_data():
        for filepath in tqdm(id_mp4s):
            # print(filepath)
            save_dir = filepath.replace(prefix, outdir)
            save_dir = save_dir.replace(".mp4", "")
            os.makedirs(save_dir, exist_ok=True)
            read_video(filepath, save_dir)
            # print(save_dir)
            # os.system(f'')
            # raise OSError("")
    run_all_data()


# screen bash -c "source /root/miniconda3/bin/activate zfacetalker && python data_mp4.py"

# ffmpeg -framerate 25 -i /data/split_video_25fps_imgs/BarackObama_1/%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p test-old.mp4
# ffmpeg -framerate 25 -i /data/split_video_25fps_imgs-2/BarackObama_1/%d_mask.jpg -c:v libx264 -y  test-new.mp4