#!/bin/bash
# Train PRnet on Sciplex dataset, FULLY ALIGNED with ours_6
# Usage: bash train_sciplex_ours6_aligned.sh [UC]

cd /mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/baselines/PRnet

# UC mode if argument is provided
UC_FLAG=""
if [ "$1" = "UC" ]; then
    UC_FLAG="--UC"
    LOG_FILE="logs/train_sciplex_ours6_aligned_UC.txt"
else
    LOG_FILE="logs/train_sciplex_ours6_aligned.txt"
fi

mkdir -p logs

srun -p medai_p --gres=gpu:1 --cpus-per-task=8 --quotatype=reserved python -u train_sciplex_ours6_aligned.py \
    --data_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed" \
    --se_inputs_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed_se_inputs_memmap/sciplex" \
    --cell_lines "A549" "K562" "MCF7" \
    --set_size 24 \
    --batch_size 1 \
    --num_workers 4 \
    --x_dimension 2000 \
    --hidden_layer_sizes 128 \
    --z_dimension 64 \
    --comb_dimension 64 \
    --drug_dimension 1024 \
    --dr_rate 0.05 \
    --num_epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-8 \
    --loss_type "gaussian" \
    --checkpoint_freq 10 \
    --eval_freq_steps 500 \
    --de_topk 50 \
    --num_eval_samples 3 \
    --save_dir "./outputs_prnet_sciplex" \
    --gpu 0 \
    $UC_FLAG \
    > $LOG_FILE 2>&1 &

echo "Training started in background. See $LOG_FILE"
