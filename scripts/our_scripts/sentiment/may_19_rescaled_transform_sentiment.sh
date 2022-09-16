RESIDUAL_METHOD=rescale_to_expert
EXPERT_SIZE=large
MODEL_DIR=models/experts/sentiment/$EXPERT_SIZE
PROMPT_TYPE=positive_prompts
PROMPTS_DATASET=prompts/sentiment_prompts-10k/$PROMPT_TYPE.jsonl
TARGET_SENTIMENT=negative
STEERING_LAYERS=( 17 27 )
TRAIN_EPOCHS=( 100 )
OUTPUT_DIR=generations/sentiment/$PROMPT_TYPE/${EXPERT_SIZE}_experts/$TARGET_SENTIMENT/

declare -a ALPHAS=(
    "1.0 5 -2.0"
    "1.0 10 -5"
    #"1.0 20 -10"
    "1.0 40 -25"
    #"0.1 10 -5"
    "0 2.0 -1.0"
    #"0 1.0 0"
    #"0 0 -1.0"
    #"0 1.0 -1.0"
    #"1.0 0 -1.0"
    #"-1.0 1.0 0"
    #"1.0 1.0 0"
    #"1.0 1.0 -1.0"
    #"-1.0 2.0 -1.0"
)

for TRAIN_EPOCH in "${TRAIN_EPOCHS[@]}"
do
    for STEERING_LAYER in "${STEERING_LAYERS[@]}"
    do
        for ALPHA_TUPLE in "${ALPHAS[@]}"
        do
            read -a strarr <<< "$ALPHA_TUPLE"
            echo ${strarr[0]} ${strarr[1]} ${strarr[2]}
            ALPHA_BASE=${strarr[0]}
            ALPHA_EXPERT=${strarr[1]}
            ALPHA_ANTIEXPERT=${strarr[2]}
            NEW_OUTPUT_DIR=$OUTPUT_DIR/${RESIDUAL_METHOD}/dexperts_steer/steering_large_gpt2/alphas_${ALPHA_BASE}_${ALPHA_EXPERT}_${ALPHA_ANTIEXPERT}/layer_${STEERING_LAYER}_freeze_emb_lm_head_with_expert_epochs${TRAIN_EPOCH}_train_val_sst/combine_at_layer/
            #echo $ALPHA_BASE
            #echo $ALPHA_EXPERT
            #echo $ALPHA_ANTIEXPERT
            #echo $NEW_OUTPUT_DIR
            python -m scripts.run_sentiment_experiment \
                --use-dataset \
                --dataset-file $PROMPTS_DATASET \
                --model-type dexperts-steer \
                --model gpt2-large \
                --pos-model $MODEL_DIR/finetuned_gpt2_train_val_sst_${TARGET_SENTIMENT}_experimental_freeze_emb_and_lmhead_layers${STEERING_LAYER}_${TRAIN_EPOCH}_maxepochs \
                --neg-model $MODEL_DIR/finetuned_gpt2_train_val_sst_positive_experimental_freeze_emb_and_lmhead_layers${STEERING_LAYER}_${TRAIN_EPOCH}_maxepochs \
                --alpha_base $ALPHA_BASE \
                --alpha_expert $ALPHA_EXPERT \
                --alpha_antiexpert $ALPHA_ANTIEXPERT \
                --filter_p 0.9 \
                --steering-layer $STEERING_LAYER \
                --batch-size 50 \
                $NEW_OUTPUT_DIR

            if [ $? -eq 0 ]; then
                python -m scripts.evaluation.evaluate_generations \
                    --generations_file $NEW_OUTPUT_DIR/prompted_gens_dexperts-steer.jsonl
            fi
        done
    done
done
