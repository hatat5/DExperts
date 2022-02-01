MODEL_NAME=gpt2-large
DATA_DIR=datasets/jigsaw-unintended-bias-in-toxicity-classification/
BATCH_SIZE=4
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=16

LAYERS_TO_STEER=( 20 21 22 23 24 25 26 27 28 29 )
for LAYER_TO_STEER in "${LAYERS_TO_STEER[@]}"
do
    python -m scripts.finetuning.finetune_gpt2 \
        --output_dir models/experts/toxicity/large/finetuned_gpt2_toxic_experimental_layers$LAYER_TO_STEER \
        --model_type gpt2 \
        --model_name_or_path $MODEL_NAME \
        --do_train \
        --num_train_epochs 1 \
        --block_size $BLOCK_SIZE \
        --save_total_limit 1 \
        --dataloader_drop_last \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --train_data_file $DATA_DIR/toxicity_gte0.5.txt \
        --overwrite_cache \
        --experimental_group experimental \
        --layers_to_finetune $LAYER_TO_STEER \
        --overwrite_output_dir
done
