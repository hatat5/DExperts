RESULTS_DIR="generations/toxicity/nontoxic_prompts_random2000/language_en/"
SECONDARY_DIR="steering_large_gpt2/alpha_2.0"
GEN_FILES=( $RESULTS_DIR"/small_experts/dexperts_anti_only/"$SECONDARY_DIR"/prompted_gens_dexperts.jsonl" $RESULTS_DIR"/small_experts/dexperts/"$SECONDARY_DIR"/prompted_gens_dexperts.jsonl" $RESULTS_DIR"/large_experts/dexperts_anti_only/"$SECONDARY_DIR"/prompted_gens_dexperts.jsonl" $RESULTS_DIR"/large_experts/dexperts/"$SECONDARY_DIR"/prompted_gens_dexperts.jsonl" $RESULTS_DIR"/large_experts/dexperts_steer_anti_only/"$SECONDARY_DIR"/layer_24+/combine_at_logit/prompted_gens_dexperts.jsonl" $RESULTS_DIR"/large_experts/dexperts_steer/"$SECONDARY_DIR"/layer_24+_with_expert/combine_at_logit/prompted_gens_dexperts.jsonl" $RESULTS_DIR"/large_experts/dexperts_steer_anti_only/"$SECONDARY_DIR"/layer_24_freeze_emb_lm_head/combine_at_layer/prompted_gens_dexperts-steer.jsonl" $RESULTS_DIR"/large_experts/dexperts_steer/"$SECONDARY_DIR"/layer_24_freeze_emb_lm_head_with_expert/combine_at_layer/prompted_gens_dexperts-steer.jsonl")

for GEN_FILE in "${GEN_FILES[@]}"
do
    echo $GEN_FILE
    python -m scripts.evaluation.evaluate_generations \
        --generations_file $GEN_FILE
done
