CHECKPOINTS=(
    "google/mt5-base"
)

for CHECKPOINT in ${CHECKPOINTS[@]}; do
    echo "checkpoint: $CHECKPOINT"
    python train_t2t_ner.py \
        --batch_size=12 \
        --num_train_epochs=10 \
        --learning_rate=2e-5 \
        --max_input_length=512 \
        --max_target_length=128 \
        --checkpoint=$CHECKPOINT \
        --output_dir=$CHECKPOINT
        # --fold=5 \
        # --cross-validation
    
done
