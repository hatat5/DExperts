API_RATE=20
EXPERT_SIZE=large
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
PROMPTS_DATASET=prompts/nontoxic_prompts_random2000.jsonl
ALPHAS=( 2.0 )

STEERING_LAYERS=( 24 )
OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/language_en/${EXPERT_SIZE}_experts/

for ALPHA in "${ALPHAS[@]}"
do
    for STEERING_LAYER in "${STEERING_LAYERS[@]}"
    do
        NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer_anti_only/steering_large_gpt2/alpha_$ALPHA/layer_${STEERING_LAYER}_freeze_emb_lm_head/combine_at_logit/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            --steering-layer $STEERING_LAYER \
            $NEW_OUTPUT_DIR
        
        NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer/steering_large_gpt2/alpha_$ALPHA/layer_${STEERING_LAYER}_freeze_emb_lm_head_with_expert/combine_at_logit/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
            --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            --steering-layer $STEERING_LAYER \
            $NEW_OUTPUT_DIR
    done
done

#EXPERT_SIZE=small
#MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
#OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/language_en/${EXPERT_SIZE}_experts/
#
#for ALPHA in "${ALPHAS[@]}"
#do
#    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_anti_only/steering_large_gpt2/alpha_$ALPHA/
#    python -m scripts.run_toxicity_experiment \
#        --use-dataset \
#        --dataset-file $PROMPTS_DATASET \
#        --model-type dexperts \
#        --model gpt2-large \
#        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
#        --perspective-rate-limit $API_RATE \
#        --alpha $ALPHA \
#        --filter_p 0.9 \
#        $NEW_OUTPUT_DIR
#    
#    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts/steering_large_gpt2/alpha_$ALPHA/
#    python -m scripts.run_toxicity_experiment \
#        --use-dataset \
#        --dataset-file $PROMPTS_DATASET \
#        --model-type dexperts \
#        --model gpt2-large \
#        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
#        --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
#        --perspective-rate-limit $API_RATE \
#        --alpha $ALPHA \
#        --filter_p 0.9 \
#        $NEW_OUTPUT_DIR
#done
#
#EXPERT_SIZE=large
#MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
#OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/language_en/${EXPERT_SIZE}_experts/
#
#for ALPHA in "${ALPHAS[@]}"
#do
#    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_anti_only/steering_large_gpt2/alpha_$ALPHA/
#    python -m scripts.run_toxicity_experiment \
#        --use-dataset \
#        --dataset-file $PROMPTS_DATASET \
#        --model-type dexperts \
#        --model gpt2-large \
#        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
#        --perspective-rate-limit $API_RATE \
#        --alpha $ALPHA \
#        --filter_p 0.9 \
#        $NEW_OUTPUT_DIR
#    
#    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts/steering_large_gpt2/alpha_$ALPHA/
#    python -m scripts.run_toxicity_experiment \
#        --use-dataset \
#        --dataset-file $PROMPTS_DATASET \
#        --model-type dexperts \
#        --model gpt2-large \
#        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
#        --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
#        --perspective-rate-limit $API_RATE \
#        --alpha $ALPHA \
#        --filter_p 0.9 \
#        $NEW_OUTPUT_DIR
#done
#
##
##ALPHA=2.0
##API_RATE=20
##EXPERT_SIZE=large
##MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
##OUTPUT_DIR=generations/toxicity/dexperts_anti-only
###OUTPUT_DIR=generations/nishant_toy_prompt/dexperts_anti-only
##
##python -m scripts.run_toxicity_experiment \
##    --use-dataset \
##    --dataset-file $PROMPTS_DATASET \
##    --model-type dexperts \
##    --model gpt2-large \
##    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
##    --perspective-rate-limit $API_RATE \
##    --alpha $ALPHA \
##    --filter_p 0.9 \
##    $OUTPUT_DIR
#
#OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/language_en/${EXPERT_SIZE}_experts/
#
#for ALPHA in "${ALPHAS[@]}"
#do
#    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer_anti_only/steering_large_gpt2/alpha_$ALPHA/layer_24+/combine_at_logit/
#    python -m scripts.run_toxicity_experiment \
#        --use-dataset \
#        --dataset-file $PROMPTS_DATASET \
#        --model-type dexperts \
#        --model gpt2-large \
#        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_layers24+ \
#        --perspective-rate-limit $API_RATE \
#        --alpha $ALPHA \
#        --filter_p 0.9 \
#        $NEW_OUTPUT_DIR
#    
#    NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_steer/steering_large_gpt2/alpha_$ALPHA/layer_24+_with_expert/combine_at_logit/
#    python -m scripts.run_toxicity_experiment \
#        --use-dataset \
#        --dataset-file $PROMPTS_DATASET \
#        --model-type dexperts \
#        --model gpt2-large \
#        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_layers24+ \
#        --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic_experimental_layers24+ \
#        --perspective-rate-limit $API_RATE \
#        --alpha $ALPHA \
#        --filter_p 0.9 \
#        $NEW_OUTPUT_DIR
#done
#
