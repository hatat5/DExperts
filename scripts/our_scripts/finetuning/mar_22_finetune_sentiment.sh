MODEL_NAME=gpt2-large
DATA_DIR=datasets/SST-5/
BATCH_SIZE=16
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=4
TRAIN_EPOCHS=3
SENTIMENTS=( positive negative )

for SENTIMENT in "${SENTIMENTS[@]}"
do
    #python -m scripts.finetuning.finetune_gpt2 \
    #    --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_freeze_emb_and_lmhead_layers24+ \
    #    --model_type gpt2 \
    #    --model_name_or_path $MODEL_NAME \
    #    --do_train \
    #    --num_train_epochs $TRAIN_EPOCHS \
    #    --block_size $BLOCK_SIZE \
    #    --save_total_limit 1 \
    #    --dataloader_drop_last \
    #    --per_device_train_batch_size $BATCH_SIZE \
    #    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    #    --train_data_file $DATA_DIR/$SENTIMENT.txt \
    #    --overwrite_cache \
    #    --experimental_group experimental \
    #    --overwrite_output_dir

    #python -m scripts.finetuning.finetune_gpt2 \
    #    --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_layers24+ \
    #    --model_type gpt2 \
    #    --model_name_or_path $MODEL_NAME \
    #    --do_train \
    #    --num_train_epochs $TRAIN_EPOCHS \
    #    --block_size $BLOCK_SIZE \
    #    --save_total_limit 1 \
    #    --dataloader_drop_last \
    #    --per_device_train_batch_size $BATCH_SIZE \
    #    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    #    --train_data_file $DATA_DIR/$SENTIMENT.txt \
    #    --overwrite_cache \
    #    --experimental_group experimental \
    #    --overwrite_output_dir \
    #    --finetune_lm_head

    #python -m scripts.finetuning.finetune_gpt2 \
    #    --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_finetune_emb_and_lmhead_layers24+ \
    #    --model_type gpt2 \
    #    --model_name_or_path $MODEL_NAME \
    #    --do_train \
    #    --num_train_epochs $TRAIN_EPOCHS \
    #    --block_size $BLOCK_SIZE \
    #    --save_total_limit 1 \
    #    --dataloader_drop_last \
    #    --per_device_train_batch_size $BATCH_SIZE \
    #    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    #    --train_data_file $DATA_DIR/$SENTIMENT.txt \
    #    --overwrite_cache \
    #    --experimental_group experimental \
    #    --overwrite_output_dir \
    #    --finetune_embedding \
    #    --finetune_lm_head

    LAYERS_TO_STEER=( 24 )
    for LAYER_TO_STEER in "${LAYERS_TO_STEER[@]}"
    do
        python -m scripts.finetuning.finetune_gpt2 \
            --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_freeze_emb_and_lmhead_layers$LAYER_TO_STEER \
            --model_type gpt2 \
            --model_name_or_path $MODEL_NAME \
            --do_train \
            --do_eval \
            --num_train_epochs 3 \
            --block_size $BLOCK_SIZE \
            --save_total_limit 1 \
            --dataloader_drop_last \
            --per_device_train_batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --train_data_file $DATA_DIR/$SENTIMENT.txt \
            --eval_data_file $DATA_DIR/$SENTIMENT.txt \
            --overwrite_cache \
            --experimental_group experimental \
            --layers_to_finetune $LAYER_TO_STEER \
            --overwrite_output_dir

        #python -m scripts.finetuning.finetune_gpt2 \
        #    --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_layers$LAYER_TO_STEER \
        #    --model_type gpt2 \
        #    --model_name_or_path $MODEL_NAME \
        #    --do_train \
        #    --num_train_epochs $TRAIN_EPOCHS \
        #    --block_size $BLOCK_SIZE \
        #    --save_total_limit 1 \
        #    --dataloader_drop_last \
        #    --per_device_train_batch_size $BATCH_SIZE \
        #    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        #    --train_data_file $DATA_DIR/$SENTIMENT.txt \
        #    --overwrite_cache \
        #    --experimental_group experimental \
        #    --layers_to_finetune $LAYER_TO_STEER \
        #    --overwrite_output_dir \
        #    --finetune_lm_head
        #
        #python -m scripts.finetuning.finetune_gpt2 \
        #    --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT}_experimental_finetune_emb_and_lmhead_layers$LAYER_TO_STEER \
        #    --model_type gpt2 \
        #    --model_name_or_path $MODEL_NAME \
        #    --do_train \
        #    --num_train_epochs $TRAIN_EPOCHS \
        #    --block_size $BLOCK_SIZE \
        #    --save_total_limit 1 \
        #    --dataloader_drop_last \
        #    --per_device_train_batch_size $BATCH_SIZE \
        #    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        #    --train_data_file $DATA_DIR/$SENTIMENT.txt \
        #    --overwrite_cache \
        #    --experimental_group experimental \
        #    --layers_to_finetune $LAYER_TO_STEER \
        #    --overwrite_output_dir \
        #    --finetune_embedding \
        #    --finetune_lm_head
    done
    
    python -m scripts.finetuning.finetune_gpt2 \
        --output_dir models/experts/sentiment/large/finetuned_gpt2_${SENTIMENT} \
        --model_type gpt2 \
        --model_name_or_path $MODEL_NAME \
        --do_train \
        --do_eval \
        --num_train_epochs $TRAIN_EPOCHS \
        --block_size $BLOCK_SIZE \
        --save_total_limit 1 \
        --dataloader_drop_last \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --train_data_file $DATA_DIR/$SENTIMENT.txt \
        --eval_data_file $DATA_DIR/$SENTIMENT.txt \
        --overwrite_cache \
        --overwrite_output_dir
    
    python -m scripts.finetuning.finetune_gpt2 \
        --output_dir models/experts/sentiment/small/finetuned_gpt2_${SENTIMENT} \
        --model_type gpt2 \
        --model_name_or_path gpt2 \
        --do_train \
        --do_eval \
        --num_train_epochs $TRAIN_EPOCHS \
        --block_size $BLOCK_SIZE \
        --save_total_limit 1 \
        --dataloader_drop_last \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --train_data_file $DATA_DIR/$SENTIMENT.txt \
        --eval_data_file $DATA_DIR/$SENTIMENT.txt \
        --overwrite_cache \
        --overwrite_output_dir
done

