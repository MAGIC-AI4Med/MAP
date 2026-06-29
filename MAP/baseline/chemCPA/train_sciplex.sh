#!/bin/bash
# Train chemCPA on Sciplex dataset, FULLY ALIGNED with ours_6
# Usage: bash train_sciplex.sh

cd /mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/baselines/chemCPA

srun -p medai_p --gres=gpu:1 --cpus-per-task=8 --quotatype=reserved python -u train_sciplex.py \
    --data_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed" \
    --se_inputs_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed_se_inputs_memmap/sciplex" \
    --cell_lines A549 K562 MCF7 \
    --set_size 24 \
    --batch_size 1 \
    --num_workers 4 \
    --embedding_model "ecfp4" \
    --embedding_path "./embeddings/ecfp4.npy" \
    --num_epochs 100 \
    --checkpoint_freq 5 \
    --eval_freq_steps 10000 \
    --save_dir "./outputs_chemCPA_sciplex" \
    --gpu 0 \
    > logs/train_sciplex.txt 2>&1 &

echo "Training started in background. See logs/train_sciplex.txt"
