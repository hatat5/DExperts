#!/bin/bash

GENERATIONS_FILE="generations/sentiment/positive_prompts/large_experts/negative/dexperts_steer_expert_only/steering_large_gpt2/alpha_1.0_beta_4.0/layer_32_freeze_emb_lm_head_epochs100_train_val_sst/combine_at_layer/prompted_gens_dexperts-steer.jsonl"
python -m scripts.evaluation.evaluate_generations \
        --generations_file $GENERATIONS_FILE

GENERATIONS_FILE="generations/sentiment/positive_prompts/large_experts/negative/dexperts_steer_expert_only/steering_large_gpt2/alpha_3.0_beta_3.0/layer_22_freeze_emb_lm_head_epochs100_train_val_sst/combine_at_layer/prompted_gens_dexperts-steer.jsonl"
python -m scripts.evaluation.evaluate_generations \
        --generations_file $GENERATIONS_FILE

GENERATIONS_FILE="generations/sentiment/positive_prompts/large_experts/negative/dexperts_steer_expert_only/steering_large_gpt2/alpha_2.0_beta_2.0/layer_22_freeze_emb_lm_head_epochs100_train_val_sst/combine_at_layer/prompted_gens_dexperts-steer.jsonl"
python -m scripts.evaluation.evaluate_generations \
        --generations_file $GENERATIONS_FILE

GENERATIONS_FILE="generations/sentiment/positive_prompts/large_experts/negative/dexperts_steer_anti_only/steering_large_gpt2/alpha_1.0_beta_4.0/layer_32_freeze_emb_lm_head_epochs100_train_val_sst/combine_at_layer/prompted_gens_dexperts-steer.jsonl"
python -m scripts.evaluation.evaluate_generations \
        --generations_file $GENERATIONS_FILE

GENERATIONS_FILE="generations/sentiment/positive_prompts/large_experts/negative/dexperts_steer/steering_large_gpt2/alpha_1.0_beta_4.0/layer_32_freeze_emb_lm_head_with_expert_epochs100_train_val_sst/combine_at_layer/prompted_gens_dexperts-steer.jsonl"
python -m scripts.evaluation.evaluate_generations \
        --generations_file $GENERATIONS_FILE
