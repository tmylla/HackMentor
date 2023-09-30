#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train/train_lora_iio.py \
  --base_model 'models/pretrined/vicuna-7b/' \
  --data_path 'data/seed_generation-14k.json' \
  --output_dir './lora_models/vicuna-7b-lora-iio' \
  --batch_size 32 \
  --micro_batch_size 2 \
  --num_epochs 8 \
  --learning_rate 2e-5 \
  --cutoff_len 1024 \
  --val_set_size 120 \
  --adapter_name lora \
