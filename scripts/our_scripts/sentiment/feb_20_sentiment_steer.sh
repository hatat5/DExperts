ALPHAS=( 3.2 )
EXPERT_SIZE=large
MODEL_DIR=models/experts/sentiment/$EXPERT_SIZE
PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl

STEERING_LAYERS=( 24 )
OUTPUT_DIR=generations/sentiment/neutral_prompts/${EXPERT_SIZE}_experts/positive/

#for ALPHA in "${ALPHAS[@]}"
#do
#    for STEERING_LAYER in "${STEERING_LAYERS[@]}"
#    do
#        NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer_anti_only/steering_large_gpt2/alpha_$ALPHA/layer_${STEERING_LAYER}_freeze_emb_lm_head/combine_at_layer/
#        python -m scripts.run_sentiment_experiment \
#            --use-dataset \
#            --dataset-file $PROMPTS_DATASET \
#            --model-type dexperts-steer \
#            --model gpt2-large \
#            --pos-model $MODEL_DIR/finetuned_gpt2_positive_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
#            --alpha $ALPHA \
#            --filter_p 0.9 \
#            --steering-layer $STEERING_LAYER \
#            $NEW_OUTPUT_DIR
#        
#        NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer/steering_large_gpt2/alpha_$ALPHA/layer_${STEERING_LAYER}_freeze_emb_lm_head_with_expert/combine_at_layer/
#        python -m scripts.run_sentiment_experiment \
#            --use-dataset \
#            --dataset-file $PROMPTS_DATASET \
#            --model-type dexperts-steer \
#            --model gpt2-large \
#            --pos-model $MODEL_DIR/finetuned_gpt2_positive_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
#            --neg-model $MODEL_DIR/finetuned_gpt2_negative_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
#            --alpha $ALPHA \
#            --filter_p 0.9 \
#            --steering-layer $STEERING_LAYER \
#            $NEW_OUTPUT_DIR
#    done
#done

for ALPHA in "${ALPHAS[@]}"
do
    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer_anti_only/steering_large_gpt2/alpha_$ALPHA/layer_24+/combine_at_logit/
    python -m scripts.run_sentiment_experiment \
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model gpt2-large \
        --pos-model $MODEL_DIR/finetuned_gpt2_positive_experimental_layers24+ \
        --alpha $ALPHA \
        --filter_p 0.9 \
        --batch-size 48 \
        $NEW_OUTPUT_DIR
    
    #NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer/steering_large_gpt2/alpha_$ALPHA/layer_24+_with_expert/combine_at_logit/
    #python -m scripts.run_sentiment_experiment \
    #    --use-dataset \
    #    --dataset-file $PROMPTS_DATASET \
    #    --model-type dexperts \
    #    --model gpt2-large \
    #    --pos-model $MODEL_DIR/finetuned_gpt2_positive_experimental_layers24+ \
    #    --neg-model $MODEL_DIR/finetuned_gpt2_negative_experimental_layers24+ \
    #    --alpha $ALPHA \
    #    --filter_p 0.9 \
    #    $NEW_OUTPUT_DIR
done

