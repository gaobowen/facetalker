export CUDA_VISIBLE_DEVICES="3,4,5,6,7"

accelerate launch --main_process_port 30001 train_unet.py \
--mixed_precision="no" \
--data_root="/data/gaobowen/split_video_25fps_sdvae320-2" \
--pretrained_model_name_or_path="/data/gaobowen/facetalker/models/musetalk/unet.pth" \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=1000000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=20 \
--output_dir="output_unet2" \
--checkpointing_steps=4000 \
--resume_from_checkpoint="latest" \
--lr_scheduler="constant" \
--finetune_rgb \
--finetune_vgg \

# nohup ./train_unet.sh > train-unet2.out 2>&1 &
# tensorboard --logdir=runs --port 9600
# conda activate facetalker