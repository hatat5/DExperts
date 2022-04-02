MODEL_DIR=models/experts/toxicity/large
for layer_num in 20 21 22 23 24 25 26 27 28 29
do
    echo $layer_num 
    cat $MODEL_DIR/finetuned_gpt2_toxic_experimental_layers$layer_num/log_history.json
done
