#!/bin/bash

set -e
set -v

BASE_DIR=`pwd`
TRAIN_PATH=data/lang8.train.ccl22
VALID_PATH=data/cged2021/test_2021.txt
BASE_MODEL=chinese-bert-wwm-ext
VOCAB_PATH=$BASE_DIR/data/output_vocabulary/
SAVE_MODEL=checkpoints/stage1
NUM_EPOCH=10
UPDATE_PER_EPOCH=0

CUDA_VISIBLE_DEVICES=3 python ./train.py \
	--train_set $TRAIN_PATH \
	--dev_set $VALID_PATH \
	--model_dir $SAVE_MODEL \
    --vocab_path $VOCAB_PATH \
	--n_epoch $NUM_EPOCH \
	--lr 1e-5 \
	--cold_steps_count 1 \
	--accumulation_size 2 \
	--updates_per_epoch $UPDATE_PER_EPOCH  \
	--tn_prob 0 \
	--tp_prob 1 \
	--transformer_model $BASE_MODEL \
	--special_tokens_fix 0 \
	--batch_size 64 \
	--pretrain_folder $BASE_MODEL \
	--patience 5 \
