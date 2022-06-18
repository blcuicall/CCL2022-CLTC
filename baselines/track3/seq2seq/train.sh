#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR=data/processed
BART_DIR=pretrained-models/bart-base-chinese
MODEL_NAME=bart-baseline
mkdir -p checkpoints/$MODEL_NAME

python train.py $DATA_DIR \
    --fp16 \
    --seed 42 \
    --user-dir bart-zh \
    --arch hf_bart \
    --task translation_hf_bart \
    --hf-model-name $BART_DIR \
    --optimizer adam \
    --clip-norm 0.1 \
    --lr 1e-5 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --update-freq 4 \
    --max-tokens 2048 \
    --max-epoch 20 \
    --num-workers 10 \
    --save-dir checkpoints/$MODEL_NAME \
    --log-format simple \
    --log-interval 50 \
    --left-pad-source \
    2>&1 | tee -a checkpoints/$MODEL_NAME/training.log
