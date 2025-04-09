export CUDA_VISIBLE_DEVICES="4,5,6,7"

accelerate launch --main_process_port 30001 train_hubert.py \
--mixed_precision="no" \
--data_root="/data/gaobowen/split_video_25fps_stable" \
--train_batch_size=320 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=1000000 \
--learning_rate=5e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=20 \
--output_dir="output_hubert" \
--checkpointing_steps=1000 \
--resume_from_checkpoint="latest" \
--lr_scheduler="constant" \

# loss 0.5 终止