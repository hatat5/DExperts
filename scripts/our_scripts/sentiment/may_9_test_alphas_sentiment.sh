ALPHA_BASES=( 1.0 )
#ALPHA_EXPERTS=( 40.0 80.0 160.0 320.0 )
ALPHA_EXPERTS=( 0 0.25 0.33 0.5 1.0 1.5 2.0 3.0 5.0 10.0 20.0 40.0 80.0 160.0 320.0 )
ALPHA_ANTIEXPERTS=( -20.0 )
#ALPHA_ANTIEXPERTS=( 0 -0.25 -0.33 -0.5 -1.0 -1.5 -2.0 -3.0 -5.0 -10.0 )
EXPERT_SIZE=large
MODEL_DIR=models/experts/sentiment/$EXPERT_SIZE
PROMPT_TYPE=positive_prompts_mini
PROMPTS_DATASET=prompts/sentiment_prompts-10k/$PROMPT_TYPE.jsonl
TARGET_SENTIMENT=negative
#STEERING_LAYERS=( 2 12 22 32 )
STEERING_LAYERS=( 12 22 )
TRAIN_EPOCHS=( 100 )
OUTPUT_DIR=generations/sentiment/$PROMPT_TYPE/${EXPERT_SIZE}_experts/$TARGET_SENTIMENT/

for TRAIN_EPOCH in "${TRAIN_EPOCHS[@]}"
do
    for ALPHA_BASE in "${ALPHA_BASES[@]}"
    do
        for ALPHA_EXPERT in "${ALPHA_EXPERTS[@]}"
        do
            for ALPHA_ANTIEXPERT in "${ALPHA_ANTIEXPERTS[@]}"
            do
                for STEERING_LAYER in "${STEERING_LAYERS[@]}"
                do
                    #NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer_anti_only/steering_large_gpt2/alpha_${ALPHA}_beta_${BETA}/layer_${STEERING_LAYER}_freeze_emb_lm_head_epochs${TRAIN_EPOCH}_train_val_sst/combine_at_layer/
                    #python -m scripts.run_sentiment_experiment \
                    #    --use-dataset \
                    #    --dataset-file $PROMPTS_DATASET \
                    #    --model-type dexperts-steer \
                    #    --model gpt2-large \
                    #    --neg-model $MODEL_DIR/finetuned_gpt2_train_val_sst_positive_experimental_freeze_emb_and_lmhead_layers${STEERING_LAYER}_${TRAIN_EPOCH}_maxepochs \
                    #    --alpha $ALPHA \
                    #    --beta $BETA \
                    #    --filter_p 0.9 \
                    #    --steering-layer $STEERING_LAYER \
                    #    --resume \
                    #    $NEW_OUTPUT_DIR
                    #
                    #NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer_expert_only/steering_large_gpt2/alpha_${ALPHA}_beta_${BETA}/layer_${STEERING_LAYER}_freeze_emb_lm_head_epochs${TRAIN_EPOCH}_train_val_sst/combine_at_layer/
                    #python -m scripts.run_sentiment_experiment \
                    #    --use-dataset \
                    #    --dataset-file $PROMPTS_DATASET \
                    #    --model-type dexperts-steer \
                    #    --model gpt2-large \
                    #    --pos-model $MODEL_DIR/finetuned_gpt2_train_val_sst_${TARGET_SENTIMENT}_experimental_freeze_emb_and_lmhead_layers${STEERING_LAYER}_${TRAIN_EPOCH}_maxepochs \
                    #    --alpha $ALPHA \
                    #    --beta $BETA \
                    #    --filter_p 0.9 \
                    #    --steering-layer $STEERING_LAYER \
                    #    $NEW_OUTPUT_DIR

                    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer/steering_large_gpt2/alphas_${ALPHA_BASE}_${ALPHA_EXPERT}_${ALPHA_ANTIEXPERT}/layer_${STEERING_LAYER}_freeze_emb_lm_head_with_expert_epochs${TRAIN_EPOCH}_train_val_sst/combine_at_layer/
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
                        $NEW_OUTPUT_DIR

                    if [ $? -eq 0 ]; then
                        python -m scripts.evaluation.evaluate_generations \
                            --generations_file $NEW_OUTPUT_DIR/prompted_gens_dexperts-steer.jsonl
                    fi
                done
            done
        done
    done
done

