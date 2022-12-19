#!/bin/bash

# Just an example of how to run the training script.

export HF_API_TOKEN="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"
BASE_MODEL="runwayml/stable-diffusion-inpainting"
RUN_NAME="inpaint_run1"
DATASET="/home/ubuntu/training_data"
N_GPU=1
N_EPOCHS=10
BATCH_SIZE=8

python3 -m torch.distributed.run --nproc_per_node=$N_GPU waifu_diffusion_inpainting/trainer/diffusers_trainer.py --model=$BASE_MODEL --run_name=$RUN_NAME --dataset=$DATASET --use_8bit_adam=True --gradient_checkpointing=True --batch_size=$BATCH_SIZE --fp16=True --image_log_steps=500 --epochs=$N_EPOCHS --resolution=512 --use_ema=True --clip_penultimate=False --save_steps 900