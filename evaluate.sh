#!/bin/bash

# bash -i evaluate.sh

CHECKPOINTS=(
    "google/mt5-base"
)

for CHECKPOINT in ${CHECKPOINTS[@]}; do
    echo "checkpoint: $CHECKPOINT"
    python evaluate_t2t_ner.py \
    --model_name_or_path=$CHECKPOINT \
    --batch_size=32 \
    --accelerator=cpu
done
