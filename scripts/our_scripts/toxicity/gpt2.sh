API_RATE=1
PROMPT_FILE=nontoxic_prompts_first50.jsonl
#OUTPUT_DIR=generations/toxicity/gpt2
#--dataset-file prompts/nontoxic_prompts-10k.jsonl \
OUTPUT_DIR=generations/toxicity/nontoxic_prompts_first50/gpt2

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file prompts/$PROMPT_FILE \
    --model-type gpt2 \
    --model gpt2-large \
    --perspective-rate-limit $API_RATE \
    --p 0.9 \
    $OUTPUT_DIR
