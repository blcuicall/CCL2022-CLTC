#!/bin/bash
detect_vocab_path="./data/vocabulary/d_tags.txt"
correct_vocab_path="./data/vocabulary/labels.txt"
train_path="/home/fxz/data/gec_data/yaclc/train_public_rand2.edits"
config_path="./deepspeed_config.json"
model_dir="./ckpt_deepspeed_amp_cold1_with_puncs"
ckpt_id="epoch-9"
save_dir="./ckpt_deepspeed_stage1cold1bz256_stage2cold0bz256_rand2"
pretrained_transformer_path="/home/fxz/data/pretrained_models/chinese-bert-wwm-ext"
mkdir $save_dir
cp $0 $save_dir
cp $config_path $save_dir
deepspeed --include localhost:$1 --master_port 42998 train.py \
    --config_path $config_path \
    --num_epochs 10 \
    --max_len 128 \
    --train_batch_size 128 \
    --accumulation_size 2 \
    --valid_batch_size 128 \
    --cold_step_count 0 \
    --lr 1e-5 \
    --cold_lr 1e-3 \
    --skip_correct 0 \
    --skip_complex 0 \
    --sub_token_mode "average" \
    --unk2keep 1 \
    --tp_prob 1 \
    --tn_prob 1 \
    --detect_vocab_path $detect_vocab_path \
    --correct_vocab_path $correct_vocab_path \
    --train_path $train_path \
    --use_cache 1 \
    --model_dir $model_dir \
    --ckpt_id $ckpt_id \
    --save_dir $save_dir \
    --pretrained_transformer_path $pretrained_transformer_path \
    2>&1 | tee ${save_dir}/train`date "+%Y%0m%0d_%T"`.log
