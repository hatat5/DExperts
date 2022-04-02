API_RATE=20
EXPERT_SIZE=small
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
#PROMPTS_DATASET=prompts/nishant_toy_prompt.jsonl
#OUTPUT_DIR=generations/nishant_toy_prompt/dexperts_steer/${EXPERT_SIZE}_experts/
PROMPTS_DATASET=prompts/nontoxic_prompts_random2000.jsonl
OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/dexperts_steer/${EXPERT_SIZE}_experts/
ALPHAS=( 2.0 )

for ALPHA in "${ALPHAS[@]}"
do
    NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/dexperts_anti_only/
    python -m scripts.run_toxicity_experiment \
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model gpt2-large \
        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
        --perspective-rate-limit $API_RATE \
        --alpha $ALPHA \
        --filter_p 0.9 \
        $NEW_OUTPUT_DIR
    
    #NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/dexperts/
    #python -m scripts.run_toxicity_experiment \
    #    --use-dataset \
    #    --dataset-file $PROMPTS_DATASET \
    #    --model-type dexperts \
    #    --model gpt2 \
    #    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #    --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
    #    --perspective-rate-limit $API_RATE \
    #    --alpha $ALPHA \
    #    --filter_p 0.9 \
    #    $NEW_OUTPUT_DIR
done

EXPERT_SIZE=large
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
#PROMPTS_DATASET=prompts/nishant_toy_prompt.jsonl
#OUTPUT_DIR=generations/nishant_toy_prompt/dexperts_steer/${EXPERT_SIZE}_experts/
PROMPTS_DATASET=prompts/nontoxic_prompts_random2000.jsonl
OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/dexperts_steer/${EXPERT_SIZE}_experts/
ALPHAS=( 2.0 )

for ALPHA in "${ALPHAS[@]}"
do
    NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/dexperts_anti_only/
    python -m scripts.run_toxicity_experiment \
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model gpt2-large \
        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
        --perspective-rate-limit $API_RATE \
        --alpha $ALPHA \
        --filter_p 0.9 \
        $NEW_OUTPUT_DIR
    
    NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/dexperts/
    python -m scripts.run_toxicity_experiment \
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model gpt2-large \
        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
        --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
        --perspective-rate-limit $API_RATE \
        --alpha $ALPHA \
        --filter_p 0.9 \
        $NEW_OUTPUT_DIR
done

#STEERING_LAYERS=( 21 22 23 24 25 26 27 28 29 )
#ALPHAS=( 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 10.0 )
#STEERING_LAYERS=( 21 25 29 )
#STEERING_LAYERS=( 21 )
#STEERING_LAYERS=( 22 24 )
STEERING_LAYERS=( 24 )
OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/dexperts_steer/${EXPERT_SIZE}_experts/

for ALPHA in "${ALPHAS[@]}"
do
    for STEERING_LAYER in "${STEERING_LAYERS[@]}"
    do
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_${STEERING_LAYER}_freeze_emb_lm_head/combine_at_layer/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts-steer \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            --steering-layer $STEERING_LAYER \
            $NEW_OUTPUT_DIR
        
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_${STEERING_LAYER}_freeze_emb_lm_head/combine_at_logit/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_freeze_emb_and_lmhead_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            $NEW_OUTPUT_DIR
        
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_${STEERING_LAYER}/combine_at_layer/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts-steer \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            --steering-layer $STEERING_LAYER \
            $NEW_OUTPUT_DIR
        
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_${STEERING_LAYER}/combine_at_logit/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            $NEW_OUTPUT_DIR
        
        
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_${STEERING_LAYER}_finetune_emb_lm_head/combine_at_layer/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts-steer \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_finetune_emb_and_lmhead_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            --steering-layer $STEERING_LAYER \
            $NEW_OUTPUT_DIR
        
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_${STEERING_LAYER}_finetune_emb_lm_head/combine_at_logit/
        python -m scripts.run_toxicity_experiment \
            --use-dataset \
            --dataset-file $PROMPTS_DATASET \
            --model-type dexperts \
            --model gpt2-large \
            --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_finetune_emb_and_lmhead_layers$STEERING_LAYER \
            --perspective-rate-limit $API_RATE \
            --alpha $ALPHA \
            --filter_p 0.9 \
            $NEW_OUTPUT_DIR
    done
done

#
#ALPHA=2.0
#API_RATE=20
#EXPERT_SIZE=large
#MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
#OUTPUT_DIR=generations/toxicity/dexperts_anti-only
##OUTPUT_DIR=generations/nishant_toy_prompt/dexperts_anti-only
#
#python -m scripts.run_toxicity_experiment \
#    --use-dataset \
#    --dataset-file $PROMPTS_DATASET \
#    --model-type dexperts \
#    --model gpt2-large \
#    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
#    --perspective-rate-limit $API_RATE \
#    --alpha $ALPHA \
#    --filter_p 0.9 \
#    $OUTPUT_DIR

OUTPUT_DIR=generations/toxicity/nontoxic_prompts_random2000/dexperts_steer/${EXPERT_SIZE}_experts/
for ALPHA in "${ALPHAS[@]}"
do
    #NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_24+_freeze_emb_lm_head/combine_at_logit/
    #python -m scripts.run_toxicity_experiment \
    #    --use-dataset \
    #    --dataset-file $PROMPTS_DATASET \
    #    --model-type dexperts \
    #    --model gpt2-large \
    #    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_freeze_emb_and_lmhead_layers24+ \
    #    --perspective-rate-limit $API_RATE \
    #    --alpha $ALPHA \
    #    --filter_p 0.9 \
    #    $NEW_OUTPUT_DIR
    
    NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_24+/combine_at_logit/
    python -m scripts.run_toxicity_experiment \
        --use-dataset \
        --dataset-file $PROMPTS_DATASET \
        --model-type dexperts \
        --model gpt2-large \
        --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_layers24+ \
        --perspective-rate-limit $API_RATE \
        --alpha $ALPHA \
        --filter_p 0.9 \
        $NEW_OUTPUT_DIR
    
    #NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_24+_finetune_emb_lm_head/combine_at_logit/
    #python -m scripts.run_toxicity_experiment \
    #    --use-dataset \
    #    --dataset-file $PROMPTS_DATASET \
    #    --model-type dexperts \
    #    --model gpt2-large \
    #    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic_experimental_finetune_emb_and_lmhead_layers24+ \
    #    --perspective-rate-limit $API_RATE \
    #    --alpha $ALPHA \
    #    --filter_p 0.9 \
    #    $NEW_OUTPUT_DIR
done
