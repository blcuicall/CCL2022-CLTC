BASE_DIR=`pwd`
PRETRAINED_MODEL_PATH=checkpoints/stage2
FINTUNED_MODEL_PATH=$BASE_DIR/output/best.th
VOCAB_PATH=data/output_vocabulary/
INPUT_FILE=data/cged2021/test_2021.txt
OUTPUT_FILE=checkpoints/stage2/cged2021.pred




CUDA_VISIBLE_DEVICES=0 python ./predict.py \
    --transformer_model $PRETRAINED_MODEL_PATH \
    --special_tokens_fix 0 \
    --iteration_count 4 \
    --model_path $FINTUNED_MODEL_PATH \
    --vocab_path $VOCAB_PATH \
    --input_file ${INPUT_FILE} \
    --output_file $OUTPUT_FILE \
    --additional_confidence 0. \
    --min_error_probability 0.
