#!/bin/bash
# Train chemCPA on NIPS dataset, FULLY ALIGNED with ours_6
# Usage: bash train_nips.sh

cd /mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/baselines/chemCPA

srun -p medai_p --gres=gpu:1 --cpus-per-task=8 --quotatype=reserved python -u train_nips.py \
    --data_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_nips/preprocessed" \
    --se_inputs_base_dir "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_nips/preprocessed_se_inputs_memmap/nips" \
    --cell_types "B cells" "Myeloid cells" "NK cells" "T cells CD4+" "T cells CD8+" "T regulatory cells" \
    --set_size 24 \
    --batch_size 1 \
    --num_workers 4 \
    --embedding_model "ecfp4" \
    --embedding_path "./embeddings/ecfp4.npy" \
    --num_epochs 100 \
    --checkpoint_freq 5 \
    --eval_freq_steps 200 \
    --save_dir "./outputs_chemCPA_nips" \
    --gpu 0 \
    > logs/train_nips.txt 2>&1 &

echo "Training started in background. See logs/train_nips.txt"
