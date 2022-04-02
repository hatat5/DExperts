DATA_DIR=datasets/jigsaw-unintended-bias-in-toxicity-classification/
BATCH_SIZE=16
GRAD_ACCUM_STEPS=4
BLOCK_SIZE=128
CHECKPOINTS_PER_EPOCH=5
TRAIN_EPOCHS=2

python -m scripts.finetuning.finetune_gpt2_dataset_size \
	--output_dir models/finetuned_gpt2_nontoxic \
	--model_type gpt2 \
	--model_name_or_path gpt2-large \
	--do_train \
	--num_train_epochs $TRAIN_EPOCHS \
	--block_size $BLOCK_SIZE \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $DATA_DIR/toxicity_eq0.txt \
