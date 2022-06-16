#! /bin/bash

model_path=checkpoints/07-ccl-lr1e-5-upfreq16-base
write_path=gen-with-beam-05-27/base
bash beam_process.sh \
    data/raw/train.uniq.src \
    ${model_path}/train.uniq.beam5.log \
    ${write_path} \
    train.beam5

bash beam_process.sh \
    multi-ref-dataset-2phase/valid/valid.yaclc-minimal.src \
    ${model_path}/valid.minimal.beam12.log \
    ${write_path} \
    valid.minimal.beam12

bash beam_process.sh \
    multi-ref-dataset-2phase/valid/valid.yaclc-fluency.src \
    ${model_path}/valid.fluency.beam12.log \
    ${write_path} \
    valid.fluency.beam12

bash beam_process.sh \
    multi-ref-dataset-2phase/test-phase1/test.yaclc-minimal.src \
    ${model_path}/test.phase1.minimal.beam12.log \
    ${write_path} \
    test.phase1.minimal.beam12

bash beam_process.sh \
    multi-ref-dataset-2phase/test-phase1/test.yaclc-fluency.src \
    ${model_path}/test.phase1.fluency.beam12.log \
    ${write_path} \
    test.phase1.fluency.beam12

bash beam_process.sh \
    multi-ref-dataset-2phase/test-phase2/test.yaclc-minimal.src \
    ${model_path}/test.phase2.minimal.beam12.log \
    ${write_path} \
    test.phase2.minimal.beam12

bash beam_process.sh \
    multi-ref-dataset-2phase/test-phase2/test.yaclc-fluency.src \
    ${model_path}/test.phase2.fluency.beam12.log \
    ${write_path} \
    test.phase2.fluency.beam12

