#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3  main.py \
        --eval \
        --train_dataset totaltext_train \
        --val_dataset totaltext_val \
        --dec_layers 6 \
        --depths 6 \
        --max_length 25 \
        --pad_rec \
        --pre_norm \
        --num_workers 0 \
        --padding_bins 0 \
        --data_root /data/code \
        --batch_size 1 \
        --lr 0.0005 \
        --resume /data//output/checkpoint.pth \
        --output_dir output \
        --visualize \
