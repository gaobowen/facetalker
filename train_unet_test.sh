export CUDA_VISIBLE_DEVICES="2,3"

accelerate launch --main_process_port 30002 --num_processes 1 train_unet_test.py \
--mixed_precision="no" \
--data_root="/data/gaobowen/split_video_25fps_sdvae320" \
--pretrained_model_name_or_path="/data/gaobowen/facetalker/models/musetalk/unet.pth" \
--train_batch_size=2 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=1000000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=20 \
--output_dir="output_unet_test" \
--checkpointing_steps=2000 \
--resume_from_checkpoint="latest" \
--lr_scheduler="constant" \
--finetune_rgb \
--finetune_vgg \

# nohup ./train_unet_test.sh > train_unet_test.out 2>&1 &
# tensorboard --logdir=runs --port 9600
# conda activate facetalker