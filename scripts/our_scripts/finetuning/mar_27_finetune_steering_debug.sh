MODEL_NAME=gpt2-large
DATA_DIR=datasets/SST-5/
BATCH_SIZE=16
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=4
TRAIN_EPOCHS=100
SENTIMENTS=( positive negative )

for SENTIMENT in "${SENTIMENTS[@]}"
do
    LAYERS_TO_STEER=( 24 )
    for LAYER_TO_STEER in "${LAYERS_TO_STEER[@]}"
    do
        python -m scripts.finetuning.finetune_gpt2 \
            --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_freeze_emb_and_lmhead_layers${LAYER_TO_STEER}_${TRAIN_EPOCHS} \
            --model_type gpt2 \
            --model_name_or_path $MODEL_NAME \
            --do_train \
            --do_eval \
            --num_train_epochs $TRAIN_EPOCHS \
            --block_size $BLOCK_SIZE \
            --save_total_limit 1 \
            --per_device_train_batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --train_data_file $DATA_DIR/train_$SENTIMENT.txt \
            --eval_data_file $DATA_DIR/validation_$SENTIMENT.txt \
            --overwrite_cache \
            --experimental_group experimental \
            --layers_to_finetune $LAYER_TO_STEER \
            --early_stopping \
            --overwrite_output_dir
    done
done

