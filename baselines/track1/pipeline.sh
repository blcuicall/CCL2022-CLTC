# Step 1. Data preprocessing
DATA_DIR=./exp_data/sighan
PRETRAIN_MODEL=./bert-base-chinese
mkdir -p $DATA_DIR


TRAIN_SRC_FILE=../../datasets/track1/sighan.train.ccl22.src
TRAIN_TRG_FILE=../../datasets/track1/sighan.train.ccl22.trg
DEV_SRC_FILE=../../datasets/track1/dev/yaclc-csc_dev.src
DEV_TRG_FILE=../../datasets/track1/dev/yaclc-csc_dev.lbl


if [ ! -f $DATA_DIR"/train.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $TRAIN_SRC_FILE \
    --target_dir $TRAIN_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/train.pkl" \
    --data_mode "para" \
    --normalize "True"
fi

if [ ! -f $DATA_DIR"/dev.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $DEV_SRC_FILE \
    --target_dir $DEV_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/dev.pkl" \
    --data_mode "lbl" \
    --normalize "True"
fi


# Step 2. Training
MODEL_DIR=./exps/sighan
CUDA_DEVICE=0
mkdir -p $MODEL_DIR/bak
cp ./pipeline.sh $MODEL_DIR/bak
cp train.py $MODEL_DIR/bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
    --pretrained_model $PRETRAIN_MODEL \
    --train_path $DATA_DIR"/train.pkl" \
    --dev_path $DATA_DIR"/dev.pkl" \
    --lbl_path $DEV_TRG_FILE \
    --save_path $MODEL_DIR \
    --batch_size 32 \
    --tie_cls_weight True \
    --tag "sighan" \
    2>&1 | tee $MODEL_DIR"/log.txt"
