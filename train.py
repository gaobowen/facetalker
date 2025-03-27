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
from accelerate import init_empty_weights
import logging
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm

import lpips

import glob
import datetime
import json
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_root", type=str, default="")
parser.add_argument("--learning_rate",type=float,default=2e-5)
parser.add_argument("--scale_lr",action="store_true",default=False)
parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ))
parser.add_argument("--lr_warmup_steps", type=int, default=0)

parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


parser.add_argument('--max_train_steps', type=int, default=100000)

parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (all devices) for the training dataloader."
    )

parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
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

parser.add_argument("--finetune_rgb", action="store_true", default=False)

parser.add_argument("--finetune_lpips", action="store_true", default=False)

args = parser.parse_args()

def train():
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir="./logs"
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )
    
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
    
    model = SanaTransformer2DModel(**config).to(accelerator.device)
    
    # model = init_transformer2d(model)
    
    if accelerator.is_main_process:
        writer = SummaryWriter('runs/experiment_1')
    
    params_to_optimize = (
        itertools.chain(model.parameters()))
    
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
        model.load_state_dict(torch.load(args.pretrained_model_name_or_path))
    
    # 减少显存占用，允许更大的 batch size
    model.enable_gradient_checkpointing()
    
    dataset = MyDataset(root_dir=args.data_root, mode='Rxxxxx', only_vae = False)
    # print(f'img count: {len(dataset)}')
    # , num_workers=8
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    
    # 最大步数/更新次数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    total_batch_size = args.train_batch_size * accelerator.num_processes
    
    if args.lr_scheduler == "constant":
        print(f"args.lr_scheduler ={args.lr_scheduler}")
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    else:
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler
        )
    
    if args.finetune_rgb:
        dc_ae = AutoencoderDC.from_pretrained("./models/vaedc/").half().to(accelerator.device).eval()
        dc_ae.requires_grad_(False)
        
    if args.finetune_lpips:
        lpips_func = lpips.LPIPS(net='vgg').to(accelerator.device).eval()
        lpips_func.requires_grad_(False)
    
    
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
    print(f'==> {global_step}/{args.max_train_steps}')
    # progress_bar = tqdm(initial=global_step, total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # 200,000 steps * 265 img + Lips finetune 100,000 steps * 265Batch
    #  50,000,000
    model.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, (input_vaes_tensor, audio_feats, real_vaes_tensor, real_imgs_tensor) in enumerate(dataloader):
            random_dataloader = True
            if not random_dataloader and args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    # progress_bar.update(1)
                    accelerator.print(f"Training skip {step}")
                    continue
            # 自动平均每步梯度
            with accelerator.accumulate(model):
                timesteps = torch.ones(args.train_batch_size).to(accelerator.device) * 0.31831
                
                def check_error():
                    assert torch.isnan(input_vaes_tensor).sum() == 0
                    assert torch.isinf(input_vaes_tensor).sum() == 0
                    assert torch.isnan(audio_feats).sum() == 0
                    assert torch.isinf(audio_feats).sum() == 0
                    assert torch.isnan(real_vaes_tensor).sum() == 0
                    assert torch.isinf(real_vaes_tensor).sum() == 0
                # print(input_vaes_tensor.shape)
                # B 32*5 10 10
                vaes_pred = model(hidden_states=input_vaes_tensor.float(), timestep=timesteps.float(), encoder_hidden_states=audio_feats.float()).sample
                def check_error2():
                    assert torch.isnan(vaes_pred).sum() == 0
                    assert torch.isinf(vaes_pred).sum() == 0
                check_error2()
                loss = 0
                # if args.finetune_rgb:
                #     vae_loss = F.l1_loss(vaes_pred.float(), real_vaes_tensor.float(), reduction="mean")
                # else:
                # vae_loss = F.mse_loss(vaes_pred.float(), real_vaes_tensor.float(), reduction="mean")
                vae_loss = F.l1_loss(vaes_pred.float(), real_vaes_tensor.float(), reduction="mean")
                
                loss += vae_loss
                # raise OSError("AA")
                
                if args.finetune_rgb:
                    vaes_pred = vaes_pred.half().reshape(-1, 32, 10, 10)
                    img_pred = dc_ae.decode(vaes_pred, return_dict=False)[0]
                    real_imgs_tensor = real_imgs_tensor.half().reshape(-1, 3, 320, 320)
                    # 加强下半脸
                    if random.random() < 0.6:
                        img_pred = img_pred[:,:,100:]
                        real_imgs_tensor = real_imgs_tensor[:,:,100:]
                    
                    
                    # rgb L1 + lpips
                    # loss_img = F.l1_loss(img_pred, real_imgs_tensor, reduction="mean")
                    # loss +=  0.5 * loss_img
                    
                    if args.finetune_lpips:
                        # # image should be RGB, IMPORTANT: normalized to [-1,1]
                        lpips_loss = lpips_func(img_pred, real_imgs_tensor).mean()
                        loss +=  0.1 * lpips_loss
                    
                accelerator.backward(loss)
                torch.cuda.empty_cache()
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(model.parameters())
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
                    # accelerator.wait_for_everyone() 报错 Some NCCL operations have failed or timed out
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

    
    '''
    MuseTalk 的训练过程在【 2 个 NVIDIA H20 GPU batchsize=256 gradient_accumulation_steps=16】上进行。model 模型最初使用 L1 损失和感知损失进行 200000 步的训练，大约需要 60 小时。
    随后，使用 Lip-Sync Loss 和 GAN loss 进行额外 100,000 步的训练，大约需要 30 小时。
    where we set λ= 0.01, μ = 0.01 and φ = 0.03 in our experiment.
    L= Lrec+λLp+μLGAN+φLsync
    
    LatentSync
    
    
    VideoMAEv2
    '''
    # todo: 初始化 def initialize_weights(self):
    # https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana.py
    
    # inference()
    
    # screen bash -c "source /root/miniconda3/bin/activate zfacetalker &&  python model.py"
    # tensorboard --logdir=runs --port 9600 --bind_all
    # conda activate facetalker
    # accelerate launch train.py
    # screen nohup ./train-stage2.sh & 关闭终端会造成进程退出，这里用 screen
    # CUDA_VISIBLE_DEVICES="0" bash ./train-stage2.sh
    