PRETRAIN_MODEL=./bert-base-chinese
DATA_DIR=./exp_data/sighan

TEST_SRC_FILE=data/test.sighan15.src
TSET_TRG_FILE=data/test.sighan15.lbl
TAG=sighan15

python ./data_preprocess.py \
--source_dir $TEST_SRC_FILE \
--target_dir $TSET_TRG_FILE \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"

MODEL_PATH=exps/sighan/sighan-epoch-3.pt
SAVE_PATH=exps/sighan/decode

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=5 python decode.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH"/"$TAG".lbl" ;

