#!/bin/bash

# bash -i evaluate.sh

conda activate htj

CHECKPOINTS=(
    "/home/work/team01/ICU_models/template-ULM_Y/checkpoints/kt-ulm-small-case1/checkpoint-7500"
    "/home/work/team01/ICU_models/template-ULM_Y/checkpoints/kt-ulm-small-case2/checkpoint-7500"
    "/home/work/team01/ICU_models/template-ULM_Y/checkpoints/kt-ulm-small-case3/checkpoint-11500"
)

for CHECKPOINT in ${CHECKPOINTS[@]}; do
    echo "checkpoint: $CHECKPOINT"
    python evaluate.py \
        --model_name_or_path=$CHE`CKPOINT \
        --max_seq_length=256 \
        --batch_size=32
done

