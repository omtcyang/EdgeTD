#!/bin/bash

bs=2
lr=0.0005
out=output
data=/data/yc

python3 -m torch.distributed.launch --master_port=35536 --nproc_per_node 2 --use_env main.py --data_root ${data} --batch_size ${bs} --lr ${lr} --output_dir ${out} \
        --train \
        --train_dataset totaltext_train \
        --val_dataset totaltext_val \
        --interm_layers \
        --dec_layers 6 \
        --depths 6 \
        --max_rec_length 25 \
        --pad_rec \
        --pre_norm \
        --num_workers 8 \
        --rotate_prob 0.3 \
        --padding_bins 0 \
        --epochs 300 \
        --warmup_epochs 5 