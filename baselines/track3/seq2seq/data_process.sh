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

# CONLL14_RAW_DIR=t5-data/conll14-noprompt/raw
# CONLL14_BPE_DIR=t5-data/conll14-noprompt/bpe
# CONLL14_PROCESSED_DIR=t5-data/conll14-noprompt/processed

# for lang in src trg; do
#     python bpe.py --load-dir $MODEL_DIR \
#         --input-file $CONLL14_RAW_DIR/test.${lang} \
#         --output-file $CONLL14_BPE_DIR/test.${lang}
# done

# python preprocess.py --user-dir huggingface \
#     --task translation_hf_t5 \
#     --source-lang src \
#     --target-lang trg \
#     --testpref $CONLL14_BPE_DIR/test \
#     --destdir $CONLL14_PROCESSED_DIR \
#     --srcdict $MODEL_DIR/dict.txt \
#     --tgtdict $MODEL_DIR/dict.txt \
#     --workers 20

#BEA19_RAW_DIR=t5-data/bea19/raw
#BEA19_BPE_DIR=t5-data/bea19/bpe
#BEA19_PROCESSED_DIR=t5-data/bea19/processed

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $BEA19_RAW_DIR/test.${lang} \
#--output-file $BEA19_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $BEA19_BPE_DIR/test \
#--destdir $BEA19_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20

#EMI2F_RAW_DIR=t5-data/gyafc-em-i2f/raw
#EMI2F_BPE_DIR=t5-data/gyafc-em-i2f/bpe
#EMI2F_PROCESSED_DIR=t5-data/gyafc-em-i2f/processed

#mkdir -p $EMI2F_BPE_DIR

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $EMI2F_RAW_DIR/test.${lang} \
#--output-file $EMI2F_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $EMI2F_BPE_DIR/test \
#--destdir $EMI2F_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20

#EMF2I_RAW_DIR=t5-data/gyafc-em-f2i/raw
#EMF2I_BPE_DIR=t5-data/gyafc-em-f2i/bpe
#EMF2I_PROCESSED_DIR=t5-data/gyafc-em-f2i/processed

#mkdir -p $EMF2I_BPE_DIR

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $EMF2I_RAW_DIR/test.${lang} \
#--output-file $EMF2I_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $EMF2I_BPE_DIR/test \
#--destdir $EMF2I_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20

#FRI2F_RAW_DIR=t5-data/gyafc-fr-i2f/raw
#FRI2F_BPE_DIR=t5-data/gyafc-fr-i2f/bpe
#FRI2F_PROCESSED_DIR=t5-data/gyafc-fr-i2f/processed

#mkdir -p $FRI2F_BPE_DIR

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $FRI2F_RAW_DIR/test.${lang} \
#--output-file $FRI2F_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $FRI2F_BPE_DIR/test \
#--destdir $FRI2F_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20

#FRF2I_RAW_DIR=t5-data/gyafc-fr-f2i/raw
#FRF2I_BPE_DIR=t5-data/gyafc-fr-f2i/bpe
#FRF2I_PROCESSED_DIR=t5-data/gyafc-fr-f2i/processed

#mkdir -p $FRF2I_BPE_DIR

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $FRF2I_RAW_DIR/test.${lang} \
#--output-file $FRF2I_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $FRF2I_BPE_DIR/test \
#--destdir $FRF2I_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20

#JFLEG_RAW_DIR=t5-data/jfleg/raw
#JFLEG_BPE_DIR=t5-data/jfleg/bpe
#JFLEG_PROCESSED_DIR=t5-data/jfleg/processed

#mkdir -p $JFLEG_BPE_DIR

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $JFLEG_RAW_DIR/test.${lang} \
#--output-file $JFLEG_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $JFLEG_BPE_DIR/test \
#--destdir $JFLEG_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20

#MQR_RAW_DIR=t5-data/mqr-noprompt/raw
#MQR_BPE_DIR=t5-data/mqr-noprompt/bpe
#MQR_PROCESSED_DIR=t5-data/mqr-noprompt/processed

#mkdir -p $MQR_BPE_DIR

#for lang in src trg; do
#python bpe.py --load-dir $MODEL_DIR \
#--input-file $MQR_RAW_DIR/test.${lang} \
#--output-file $MQR_BPE_DIR/test.${lang}
#done

#python preprocess.py --user-dir huggingface \
#--task translation_hf_t5 \
#--source-lang src \
#--target-lang trg \
#--testpref $MQR_BPE_DIR/test \
#--destdir $MQR_PROCESSED_DIR \
#--srcdict $MODEL_DIR/dict.txt \
#--tgtdict $MODEL_DIR/dict.txt \
#--workers 20
