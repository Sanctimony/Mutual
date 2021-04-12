# Mutual

Train the model and predict on the dev dataset:

python run_MDFN.py --data_dir mutual --model_name_or_path google/electra-large-discriminator --model_type electra --task_name mutual --output_dir output_mutual_electra --cache_dir cached_models --max_seq_length 256 --do_train --do_eval --train_batch_size 2 --eval_batch_size 2 --learning_rate 4e-6 --num_train_epochs 3 --gradient_accumulation_steps 1 --local_rank -1
