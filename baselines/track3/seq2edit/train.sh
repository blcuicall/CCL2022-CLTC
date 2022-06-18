#!/bin/bash
base_dir="/home/username/workspace"
data_dir="${base_dir}/yaclc_bake_off"
detect_vocab_path="./data/vocabulary/d_tags.txt"
correct_vocab_path="./data/vocabulary/labels.txt"
train_path="${data_dir}/train/lang8.train.ccl22.edits"
valid_path="${data_dir}/valid/yaclc-minimal_dev.edits"
config_path="./deepspeed_config.json"
timestamp=`date "+%Y%0m%0d_%T"`
save_dir="../ckpts/ckpt_$timestamp"
pretrained_transformer_path="${base_dir}/chinese-bert-wwm-ext"
mkdir $save_dir
cp $0 $save_dir
cp $config_path $save_dir
NCCL_DEBUG=INFO deepspeed --include localhost:$1 --master_port 42997 train.py \
    --config_path $config_path \
    --num_epochs 20 \
    --max_len 128 \
    --train_batch_size 128 \
    --accumulation_size 1 \
    --valid_batch_size 128 \
    --cold_step_count 2 \
    --lr 1e-5 \
    --cold_lr 1e-3 \
    --skip_correct 0 \
    --skip_complex 0 \
    --sub_token_mode "average" \
    --special_tokens_fix 1 \
    --unk2keep 0 \
    --tp_prob 1 \
    --tn_prob 1 \
    --detect_vocab_path $detect_vocab_path \
    --correct_vocab_path $correct_vocab_path \
    --do_eval \
    --train_path $train_path \
    --valid_path $valid_path \
    --use_cache 1 \
    --save_dir $save_dir \
    --pretrained_transformer_path $pretrained_transformer_path \
    --amp \
    2>&1 | tee ${save_dir}/train`date "+%Y%0m%0d_%T"`.log
