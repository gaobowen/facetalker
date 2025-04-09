export CUDA_VISIBLE_DEVICES="6,7"

accelerate launch --main_process_port 30001 train_hubert.py \
--mixed_precision="no" \
--data_root="/data/gaobowen/split_video_25fps_stable" \
--train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing \
--max_train_steps=1000000 \
--learning_rate=2e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=20 \
--output_dir="output_hubert" \
--checkpointing_steps=1000 \
--resume_from_checkpoint="latest" \
--lr_scheduler="constant" \
--finetune_rgb \
--finetune_lpips \


# nohup ./train-hubert-stage2.sh > train-hubert.out 2>&1 &
# tensorboard --logdir=runs --port 9600
# conda activate facetalker