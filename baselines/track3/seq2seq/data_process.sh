#! /bin/bash
set -e

MODEL_DIR=pretrained-models/bart-large-chinese

RAW_DIR=data/raw
BPE_DIR=data/bpe
PROCESSED_DIR=data/processed

for split in train valid; do
    for lang in src tgt; do
        python bpe.py --load-dir $MODEL_DIR \
            --input-file $RAW_DIR/${split}.${lang} \
            --output-file $BPE_DIR/${split}.${lang}
    done
done

python preprocess.py --user-dir bart-zh \
    --task translation_hf_bart \
    --source-lang src \
    --target-lang tgt \
    --trainpref $BPE_DIR/train \
    --validpref $BPE_DIR/valid \
    --destdir $PROCESSED_DIR \
    --srcdict $MODEL_DIR/dict.txt \
    --tgtdict $MODEL_DIR/dict.txt \
    --workers 20
