#!/bin/bash
# Train chemCPA on Tahoe dataset, FULLY ALIGNED with ours_6

cd /mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/baselines/chemCPA

srun -p medai_p --gres=gpu:1 --cpus-per-task=8 --quotatype=reserved python -u train_tahoe.py \
    --data_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3/preprocessed" \
    --se_inputs_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_4/preprocessed_se_inputs_memmap" \
    --cell_lines CVCL_0023 CVCL_0480 CVCL_0069 CVCL_0131 CVCL_1098 CVCL_1056 \
    --set_size 24 \
    --batch_size 1 \
    --num_workers 4 \
    --embedding_model "ecfp4" \
    --embedding_path "./embeddings/ecfp4.npy" \
    --num_epochs 100 \
    --checkpoint_freq 5 \
    --eval_freq_steps 200 \
    --save_dir "./outputs_chemCPA_ours6" \
    --gpu 0 \
    > logs/train_tahoe.txt 2>&1 &

echo "Training started in background. See logs/train_tahoe.txt"
