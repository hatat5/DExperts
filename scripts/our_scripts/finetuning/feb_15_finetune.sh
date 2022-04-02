
# This hasn't been run because it doesnt make sense
MODEL_NAME=gpt2
DATA_DIR=datasets/jigsaw-unintended-bias-in-toxicity-classification/
BATCH_SIZE=16
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=4

TRAIN_EPOCHS=2

LAYERS_TO_STEER=( 7 )
for LAYER_TO_STEER in "${LAYERS_TO_STEER[@]}"
do
    python -m scripts.finetuning.finetune_gpt2 \
        --output_dir models/experts/toxicity/small/finetuned_gpt2_toxic_experimental_freeze_emb_and_lmhead_layers$LAYER_TO_STEER \
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

    python -m scripts.finetuning.finetune_gpt2 \
        --output_dir models/experts/toxicity/small/finetuned_gpt2_nontoxic_experimental_freeze_emb_and_lmhead_layers$LAYER_TO_STEER \
        --model_type gpt2 \
        --model_name_or_path $MODEL_NAME \
        --do_train \
        --num_train_epochs $TRAIN_EPOCHS \
        --block_size $BLOCK_SIZE \
        --save_total_limit 1 \
        --dataloader_drop_last \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --train_data_file $DATA_DIR/toxicity_eq0.txt \
        --overwrite_cache \
        --experimental_group experimental \
        --layers_to_finetune $LAYER_TO_STEER \
        --overwrite_output_dir
done
