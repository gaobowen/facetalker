import os
import argparse
import itertools
import math
import random
from pathlib import Path

from diffusers import AutoencoderDC, SanaTransformer2DModel

from dataloader import MyDataset

import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from diffusers.optimization import get_scheduler
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import lpips
from tqdm import tqdm

import glob
import datetime
import json
from PIL import Image


logger = get_logger(__name__)


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--learning_rate",type=float,default=5e-5)
parser.add_argument("--scale_lr",action="store_true",default=False)
parser.add_argument("--lr_scheduler",type=str,default="cosine")
parser.add_argument("--lr_warmup_steps", type=int, default=0)

parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


parser.add_argument('--max_train_steps', type=int, default=100000)

parser.add_argument('--max_train_data', type=int, default=25600000)

parser.add_argument(
        "--train_batch_size", type=int, default=256, help="Batch size (all devices) for the training dataloader."
    )

parser.add_argument(
        "--device_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

# parser.add_argument('--max_data', type=int, default=20000000)
# parser.add_argument('--batch_size', type=int, default=10)
# parser.add_argument('--batch_size_optimizer', type=int, default=250)
# # parser.add_argument("--finetune_lips", action="store_true", default=False)

parser.add_argument("--pretrained_model_name_or_path", type=str, default="unet_epoch5.pth")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--logging_dir", type=str, default="logs",
    help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    ),)
# tensorboard --logdir=output/runs --port 9600 --bind_all

parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )

parser.add_argument("--checkpoints_total_limit", type=int, default=None)

parser.add_argument("--finetune_lips", action="store_true", default=True)


args = parser.parse_args()



def train():
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

    unet = SanaTransformer2DModel(**config).to(device)
    
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )
    
    args.gradient_accumulation_steps = args.train_batch_size // args.device_batch_size
    
    # 总共需要训练 max_train_data = 25,600,000 条数据
    # 这里的 max_train_steps 换算城 max_train_update_steps
    args.max_train_steps = args.max_train_data // (args.device_batch_size * args.gradient_accumulation_steps)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )
    
    if accelerator.is_main_process:
        writer = SummaryWriter('runs/experiment_1')
    
    params_to_optimize = (
        itertools.chain(unet.parameters()))
    
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    if args.pretrained_model_name_or_path is not None:
        unet.load_state_dict(torch.load(args.pretrained_model_name_or_path))
    else:
      
        
    unet.enable_gradient_checkpointing()
        
    dataset = MyDataset(root_dir='/data/gaobowen/split_video_25fps_imgs', resized_img=384)
    # print(f'img count: {len(dataset)}')
    # , num_workers=8
    dataloader = DataLoader(dataset, batch_size=args.device_batch_size, shuffle=True, num_workers=8)
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    
    # 最大步数/更新次数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    total_batch_size = args.train_batch_size * accelerator.num_processes
    
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(accelerator.device).eval()
    dc_ae.requires_grad_(False)
    
    loss_lpips = lpips.LPIPS(net='vgg').to(accelerator.device).eval()
    loss_lpips.requires_grad_(False)
    
    
    print(f"  Num batches each epoch = {len(dataloader)}")
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # 这里的 global_step = global_update_step
    global_step = 0
    first_epoch = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    # 这里是梯度频次
    print(f'===========> {global_step}{args.max_train_steps}')
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # 200,000 steps * 265 img + Lips finetune 100,000 steps * 265Batch
    #  50,000,000
    batch_count = 0
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, (vae_ref_t1_t2, hu_t0_t3, img1, img2, vae1, vae2) in enumerate(dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    continue
            with accelerator.accumulate(unet):
                timesteps = torch.tensor([0], device=accelerator.device)
                # B 8 48 48
                vae_pred = unet(vae_ref_t1_t2, timesteps, encoder_hidden_states=hu_t0_t3).sample
                vae1_pred = vae_pred[:,0:4]
                vae2_pred = vae_pred[:,4:]
                
                loss1 = F.l1_loss(vae1_pred.float(), vae1.float(), reduction="mean")
                loss2 = F.l1_loss(vae2_pred.float(), vae2.float(), reduction="mean")
                loss = (loss1 + loss2) * 0.5
                
                img1_pred = vae.decode_latents_tensor(vae1_pred).float()
                img2_pred = vae.decode_latents_tensor(vae2_pred).float()
                # vae_t1_t2 = vae_ref_t1_t2[:,4:]
                
                if args.finetune_lips and step % 2 == 1:
                    # 取下半张脸
                    img1_pred = img1_pred[:,:,128:]
                    img2_pred = img2_pred[:,:,128:]
                    img1 = img1[:,:,128:]
                    img2 = img2[:,:,128:]
                    
                
                # print(vae_t1_t2.shape)
                # 直接使用 rgb L1 + lpips + move
                loss_img1 = F.l1_loss(img1_pred, img1, reduction="mean")
                loss_img2 = F.l1_loss(img2_pred, img2, reduction="mean")
                
                loss +=  0.5 * (loss_img1 + loss_img2)
                
                # image should be RGB, IMPORTANT: normalized to [-1,1]
                lpips1 = loss_lpips(img1_pred, img1).mean()
                lpips2 = loss_lpips(img2_pred, img2).mean()
                lpips_coffe = 0.05 + (step % 10 / 10.0 * 0.05)
                loss +=  0.05 * (lpips1 + lpips2)
                
                loss_move = F.l1_loss(img2_pred-img1_pred, img2-img1, reduction="mean") # 运动损失
                loss +=  0.01 * loss_move
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    # if accelerator.is_main_process:
                    #     # 打印为 gradient_accumulation_steps 的倍数
                    #     print(f'=======>step {step}') 
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # # 模拟大批次, 
                # accelerator.sync_gradients 每隔 gradient_accumulation_steps 触发一次
                # 这里不需要 手动取余更新
                # if (step) % args.gradient_accumulation_steps == 0:
                #     optimizer.step()
                #     lr_scheduler.step()
                #     optimizer.zero_grad()
                
            
            # 更新频率为 args.gradient_accumulation_steps
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if global_step % 10 == 0 and accelerator.is_main_process:
                    writer.add_scalar('Training Loss', loss.item(), global_step)
    
        # end of epoch
        accelerator.wait_for_everyone()
    
    # end of train
    accelerator.end_training()



if __name__ == '__main__':
    train()
    # 卡尔曼滤波（中值滤波） 稳定人脸数据集
    # https://blog.51cto.com/u_16213379/12991281
    
    '''
    
    MuseTalk 的训练过程在 2 个 NVIDIA H20 GPU 上进行。Unet 模型最初使用 L1 损失和感知损失进行 200000 步的训练，大约需要 60 小时。
    随后，使用 Lip-Sync Loss 和 GAN loss 进行额外 100,000 步的训练，大约需要 30 小时。
    where we set λ= 0.01, μ = 0.01 and φ = 0.03 in our experiment.
    L= Lrec+λLp+μLGAN+φLsync
    '''
    
    # inference()
    
    # screen bash -c "source /root/miniconda3/bin/activate zfacetalker &&  python model.py"
    # tensorboard --logdir=runs --port 9600 --bind_all
    # conda activate zfacetalker
    # accelerate launch train.py
    # nohup ./train.sh &
    # CUDA_VISIBLE_DEVICES="0,1"
    