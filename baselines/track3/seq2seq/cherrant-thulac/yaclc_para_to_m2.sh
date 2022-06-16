#! /bin/bash

DATASET=../multi-ref-dataset-2phase

python parallel_to_m2.py \
    -f $DATASET/train/train.cherrant.para \
    -o $DATASET/train/train.char.m2 \
    -g char -w 10 &

python parallel_to_m2.py \
    -f $DATASET/valid/valid.yaclc-minimal.para \
    -o $DATASET/valid/valid.yaclc-minimal.char.m2 \
    -g char -w 10 &

python parallel_to_m2.py \
    -f $DATASET/valid/valid.yaclc-fluency.para \
    -o $DATASET/valid/valid.yaclc-fluency.char.m2 \
    -g char -w 10 &

python parallel_to_m2.py \
    -f $DATASET/test-phase1/test.yaclc-minimal.para \
    -o $DATASET/test-phase1/test.yaclc-minimal.char.m2 \
    -g char -w 10 &

python parallel_to_m2.py \
    -f $DATASET/test-phase1/test.yaclc-fluency.para \
    -o $DATASET/test-phase1/test.yaclc-fluency.char.m2 \
    -g char -w 10 &

python parallel_to_m2.py \
    -f $DATASET/test-phase2/test.yaclc-minimal.para \
    -o $DATASET/test-phase2/test.yaclc-minimal.char.m2 \
    -g char -w 10 &

python parallel_to_m2.py \
    -f $DATASET/test-phase2/test.yaclc-fluency.para \
    -o $DATASET/test-phase2/test.yaclc-fluency.char.m2 \
    -g char -w 10 &
