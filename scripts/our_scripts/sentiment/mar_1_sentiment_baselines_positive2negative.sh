ALPHA=3.2
EXPERT_SIZE=large
MODEL_DIR=models/experts/sentiment/$EXPERT_SIZE
PROMPT_TYPE=positive_prompts
TARGET_SENTIMENT=negative
PROMPTS_DATASET=prompts/sentiment_prompts-10k/$PROMPT_TYPE.jsonl
OUTPUT_DIR=generations/sentiment/$PROMPT_TYPE/${EXPERT_SIZE}_experts/$TARGET_SENTIMENT/

NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts/
python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --pos-model $MODEL_DIR/finetuned_gpt2_negative \
    --neg-model $MODEL_DIR/finetuned_gpt2_positive \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $NEW_OUTPUT_DIR

NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_anti_only/
python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --neg-model $MODEL_DIR/finetuned_gpt2_positive \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $NEW_OUTPUT_DIR

EXPERT_SIZE=small
MODEL_DIR=models/experts/sentiment/$EXPERT_SIZE
OUTPUT_DIR=generations/sentiment/$PROMPT_TYPE/${EXPERT_SIZE}_experts/$TARGET_SENTIMENT/

NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts/
python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --pos-model $MODEL_DIR/finetuned_gpt2_negative \
    --neg-model $MODEL_DIR/finetuned_gpt2_positive \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $NEW_OUTPUT_DIR

NEW_OUTPUT_DIR=$OUTPUT_DIR/dexperts_anti_only/
python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --neg-model $MODEL_DIR/finetuned_gpt2_positive \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $NEW_OUTPUT_DIR

