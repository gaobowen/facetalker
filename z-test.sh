export CUDA_VISIBLE_DEVICES="4,5,6,7"

accelerate launch --main_process_port 29509 vae.py