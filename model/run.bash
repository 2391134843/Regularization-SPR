#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python3 learn.py \
    --dataset WN18RR \
    --model CP \
    --rank 3000 \
    --optimizer Adagrad \
    --learning_rate 1e-1 \
    --batch_size 4000 \
    --regularizer SPR \
    --reg 1.1e-1 \
    --max_epochs 1000 \
    --valid 50 \
    -train \
    -id 0 \
    -save \
    -weight
