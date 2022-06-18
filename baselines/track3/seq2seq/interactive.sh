#! /usr/bin
export CUDA_VISIBLE_DEVICES=0

DATA_SET=data/processed
MODEL_PATH=$1
DATA_PATH=$2

cat ${DATA_PATH} \
    | python interactive.py $DATA_SET \
    --user-dir bart-zh \
    --task translation_hf_bart \
    --path $MODEL_PATH/checkpoint_best.pt \
    --beam 5 \
    --left-pad-source \
    --buffer-size 64 \
    --batch-size 64 \
    2>&1 | tee $MODEL_PATH/${SPLIT}.yaclc-${STYLE}.log

