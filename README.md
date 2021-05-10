# Mutual

Two ways to inject new data.
1. Use the pretraining framework showcased in [Injecting Numerical Reasoning Skills into Language Models](https://arxiv.org/pdf/2004.04487.pdf).

These were the commands used. 
python injecting_numeracy-master\pre_training\numeric_data_generation\gen_numeric_data.py
--num_samples 1e5 --num_dev_samples 1e4 --output_jsonl ./newdata/synthetic_numeric.jsonl

python injecting_numeracy-master\textual_data_generation\generate_examples.py --output_file_base ./newdata/synthetic_text

python injecting_numeracy-master\pre_training\convert_synthetic_numeric_to_drop.py --data_jsonl ./newdata/synthetic_numeric.jsonl

python injecting_numeracy-master\pre_training\convert_synthetic_texual_to_drop.py --data_json ./newdata/synthetic_text_train.json

python injecting_numeracy-master\pre_training\convert_synthetic_texual_to_drop.py --data_json ./newdata/synthetic_text_dev.json

python injecting_numeracy-master\pre_training\gen_bert\create_examples_n_features.py --split train --drop_json ./newdata/synthetic_text_train_drop_format.json --output_dir ./newdata/examples_n_features_syntext --max_seq_length 160 --max_n_samples -1

python injecting_numeracy-master\pre_training\gen_bert\create_examples_n_features.py --split dev --drop_json ./newdata/synthetic_text_dev_drop_format.json --output_dir ./newdata/examples_n_features_syntext --max_seq_length 160 --max_n_samples -1

python injecting_numeracy-master\pre_training\gen_bert\create_examples_n_features.py --split train --drop_json ./newdata/synthetic_numeric_train_drop_format.json --output_dir ./newdata/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11 --max_n_samples -1

python injecting_numeracy-master\pre_training\gen_bert\finetune_on_drop.py
--do_train --examples_n_features_dir ./newdata/examples_n_features_syntext --train_batch_size 24 --mlm_batch_size -1 --learning_rate 1e-5  --max_seq_length 160 --num_train_epochs 5.0 --warmup_proportion 0.1 --output_dir ./out/out_syntext_finetune_bert --random_shift --num_train_samples -1

To run the model:
python run_MDFN.py --data_dir mutual --model_name_or_path ./out/out_syntext_finetune_bert --model_type bert --task_name mutual --output_dir output_mutual_bert --cache_dir cached_models --max_seq_length 256 --do_train --do_eval --train_batch_size 2 --eval_batch_size 2 --learning_rate 4e-6 --num_train_epochs 3 --gradient_accumulation_steps 1 --local_rank -1


2. Generate data in MuTual format and add to training dataset with MuTual.
python injecting_numeracy-master\textual_data_generation\generate_examples.py --output_file_base ./newdata/synthetic_text

python injecting_numeracy-master\textual_data_generation\convert_synthetic_texual_to_mutual.py --data_json ./newdata/synthetic_text_train.json --save_path ./data/train

To run the model:
python run_MDFN.py --data_dir mutual --model_name_or_path google/electra-large-discriminator --model_type electra --task_name mutual --output_dir output_mutual_electra --cache_dir cached_models --max_seq_length 256 --do_train --do_eval --train_batch_size 2 --eval_batch_size 2 --learning_rate 4e-6 --num_train_epochs 3 --gradient_accumulation_steps 1 --local_rank -1