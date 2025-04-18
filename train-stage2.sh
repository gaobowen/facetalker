export CUDA_VISIBLE_DEVICES="7"


accelerate launch train.py \
--mixed_precision="no" \
--data_root="/data/gaobowen/split_video_25fps_stable_320" \
--train_batch_size=2 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing \
--max_train_steps=1000000 \
--learning_rate=2e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=0 \
--output_dir="output320" \
--checkpointing_steps=1000 \
--resume_from_checkpoint="latest" \
--lr_scheduler="constant" \
--finetune_rgb \
--finetune_lpips 

# 前期大batch 噪声小 大学习率 加快收敛，后期 小batch 小学习率 增强泛化 
# export CUDA_VISIBLE_DEVICES="0" bash ./train-stage2.sh
# nohup ./train-stage2.sh &
# tensorboard --logdir=runs --port 9600
# conda activate facetalker