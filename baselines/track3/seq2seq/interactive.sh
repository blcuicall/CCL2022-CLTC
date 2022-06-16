#! /usr/bin
export CUDA_VISIBLE_DEVICES=0

DATA_SET=data/processed
MODEL_PATH=$1
SPLIT=$2
STYLE=$3

cat data/bpe/${SPLIT}.yaclc-${STYLE}.src \
    | python interactive.py $DATA_SET \
    --user-dir bart-zh \
    --task translation_hf_bart \
    --path $MODEL_PATH/checkpoint_best.pt \
    --beam 5 \
    --left-pad-source \
    --buffer-size 64 \
    --batch-size 64 \
    2>&1 | tee $MODEL_PATH/${SPLIT}.yaclc-${STYLE}.log

