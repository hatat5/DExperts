EXPERT_SIZE=large
API_RATE=20
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
PROMPTS_DATASET=prompts/nishant_toy_prompt.jsonl
OUTPUT_DIR=generations/nishant_toy_prompt/dexperts/${EXPERT_SIZE}_experts/

ALPHAS=( 2.0 )
#STEERING_LAYERS=( 21 22 23 24 25 26 27 28 29 )
#ALPHAS=( 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 10.0 )
STEERING_LAYERS=( 21 )
for ALPHA in "${ALPHAS[@]}"
do
    for STEERING_LAYER in "${STEERING_LAYERS[@]}"
    do
        NEW_OUTPUT_DIR=$OUTPUT_DIR/alpha_$ALPHA/layer_$STEERING_LAYER/
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
    done
done
