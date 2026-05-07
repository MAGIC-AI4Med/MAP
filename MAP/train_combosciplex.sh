#!/bin/bash
# =========================
# Train ComboSciplex with DDP
# Usage: bash train_combosciplex.sh [scheme]
#   scheme: "avg_emb" (default) or "two_tokens"
# =========================
set -e

# Scheme selection: avg_emb or two_tokens
SCHEME=${1:-avg_emb}

# Experiment name
EXP_NAME="combosciplex_${SCHEME}_$(date +%Y%m%d_%H%M%S)"

# Paths
DATA_BASE_DIR="/path/to/your/dataset_combosciplex/preprocessed"
SE_INPUTS_BASE_DIR="/path/to/your/dataset_combosciplex/preprocessed"
SE_CONFIG="/path/to/your/se600m.yaml"
SE_CKPT="/path/to/your/se600m.safetensors"
MASTER_PORT=$((20000 + RANDOM % 40000))

echo "================================================"
echo "Training ComboSciplex with scheme: ${SCHEME^^}"
echo "Experiment name: $EXP_NAME"
echo "================================================"

# DDP training
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT train_combosciplex.py \
    --exp_name "$EXP_NAME" \
    --scheme "$SCHEME" \
    --data_base_dir "$DATA_BASE_DIR" \
    --se_inputs_base_dir "$SE_INPUTS_BASE_DIR" \
    --cell_lines "K562" \
    --set_size 24 \
    --batch_size 1 \
    --lr 1e-5 \
    --hvg_loss_weight 0.1 \
    --epochs 100 \
    --num_warmup_steps 30 \
    --se_config "$SE_CONFIG" \
    --se_ckpt "$SE_CKPT" \
    --amp \
    --amp_dtype bf16 \
    --num_workers 8 \
    --eval_every_steps 200 \
    --save_every_k_epochs 3" \
    > logs/train_combosciplex_${SCHEME}.log
