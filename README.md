
# Phrase-level Attention
This branch includes the code that imeplements the phrase-level attention mechanism introduced in [Phrase-Based Attentions](https://arxiv.org/abs/1810.03444) on top of the original MDFN paper [Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue](https://arxiv.org/pdf/2009.06504.pdf)

Run the following command to reproduce the reported results: 

```
python MDFN/run_MDFN.py \
--data_dir MuTual/data/mutual \
--model_name_or_path \
google/electra-large-discriminator \
--model_type electra \
--task_name mutual \
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--learning_rate 4e-6 \
--num_train_epochs 6 \
--gradient_accumulation_steps 1 \
--local_rank -1
```


