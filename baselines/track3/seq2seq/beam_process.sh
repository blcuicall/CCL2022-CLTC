#! /bin/bash
set -e

src_path=$1
log_path=$2
out_dir=$3
prefix=$4

cat ${log_path} \
    | grep ^H \
    | awk -F '\t' 'BEGIN {ORS=""} {if ($1 != prev_src) {print "\n"$3;} else {print "\t"$3} prev_src=$1} END {printf "\n"}' \
    | sed '1d' \
    > ${out_dir}/gen.trg

paste ${src_path} ${out_dir}/gen.trg \
    | awk -F '\t' '{print NR-1"\t"$0}' \
    | sed -r 's/ ##//g' \
    | sed -r 's/ //g' \
    | sed -r 's/\[UNK\]//g' \
    > ${out_dir}/${prefix}.para

rm ${out_dir}/gen.trg

