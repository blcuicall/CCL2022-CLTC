export CUDA_VISIBLE_DEVICES="0,1,2,3"
python train.py --outdir ./save_model \
--train_path ./data_qe/train.json \
--valid_path ./data_qe/valid.fluency.json \
--model_name_or_path bert-base-chinese
