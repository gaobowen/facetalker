# export CUDA_VISIBLE_DEVICES="0"

accelerate launch train.py \
--mixed_precision="no" \
--data_root="/data/gaobowen/split_video_25fps_imgs-2" \
--train_batch_size=480 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=100000 \
--learning_rate=5e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=0 \
--output_dir="output" \
--checkpointing_steps=2000 \
--resume_from_checkpoint="latest" \
--lr_scheduler="constant" \

