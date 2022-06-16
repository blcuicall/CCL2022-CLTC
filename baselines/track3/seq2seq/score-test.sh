#! /bin/bash
set -e

model_path=$1
mkdir -p ${model_path}/scores
split=$2
phase=$3
style=$4
src=multi-ref-dataset-2phase/${split}-${phase}/${split}.yaclc-${style}.src
hyp_para=${model_path}/scores/${split}.${phase}.${style}.hyp.para
hyp_m2=${model_path}/scores/${split}.${phase}.${style}.hyp.m2
ref_m2=multi-ref-dataset-2phase/${split}-${phase}/${split}.yaclc-${style}.char.m2

cat ${model_path}/${split}.${phase}.yaclc-${style}.log \
    | grep ^H \
    | sed 's/^..//' \
    | sort -n \
    | cut -f 3 \
    | sed -r 's/ ##//g' \
    | sed -r 's/ //g' \
    | sed -r 's/\[UNK\]//g' \
    > ${model_path}/scores/${split}.${phase}.${style}.hyp.trg
    #| sed -r 's/([^a-zA-Z0-9]) ([^a-zA-Z0-9])/\1\2/g' \
    #| sed -r 's/([^a-zA-Z0-9]) ([^a-zA-Z0-9])/\1\2/g' \

paste ${src} ${model_path}/scores/${split}.${phase}.${style}.hyp.trg \
    | awk '{print NR"\t"$0}' \
    > ${hyp_para}

cd cherrent-thulac
python parallel_to_m2.py \
    -f ../${hyp_para} \
    -o ../${hyp_m2} \
    -g char

python compare_m2_for_evaluation.py \
    -hyp ../${hyp_m2} \
    -ref ../${ref_m2} \
    | tee ../${model_path}/scores/score.${split}.${style}.txt
cd -

