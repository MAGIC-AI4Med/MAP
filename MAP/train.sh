#!/bin/bash

torchrun --nproc_per_node=8 train.py \
    --exp_name "train" \
    --data_base_dir "path/to/your/preprocessed/data/directory" \
    --se_inputs_base_dir "path/to/your/preprocessed/SE/inputs" \
    --cell_lines "cell line ids" \
    --batch_size 1 \
    --num_workers 4 \
    --set_size 24 \
    --epochs 100 \
    --lr 1e-5 \
    --hvg_loss_weight 0.1 \
    --smile_encoder 'MAP-KG' \
    --se_config "configs/se600m.yaml" \
    --se_ckpt "checkpoints/se600m.safetensors" \
    --amp \
    --amp_dtype "bf16" \
    --gradient_accumulation_steps 2 \
    > ./logs/train.txt