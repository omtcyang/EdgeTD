#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python3  main.py \
        --train\
        --train_dataset totaltext_train \
        --val_dataset totaltext_val \
        --batch_size 1 \
        --lr 0.0005 \
        --output_dir output \
        --data_root /mnt/TCSVT2024/ \
        --interm_layers \
        --max_rec_length 25 \
        --pad_rec \
        --pre_norm \
        --num_workers 8 \
        --rotate_prob 0.3 \
        --padding_bins 0 \
        --epochs 300 \
        --warmup_epochs 5
