# Codes for pre-train T5 on clinical corpus
Repository for
1. Train a new tokenizer on clinical corpus from the old T5 tokenizer
2. Pre-train T5 model on clinical corpus with Gap Span/Sentence Generation as the pre-training objective


## Requirements
- python 3.9.16
- pytorch 1.13.1
- CUDA 11.6


## Train new tokenizer
To train the new tokenizer from the old tokenizer, run below command.

```
# train command
python src/train_tokenizer.py --corpus_path PATH-TO-CORPUS --old_tokenizer NAME-OF-OLD-TOKENIZER

# for example
python src/train_tokenizer.py --corpus_path "Datasets/train_example.json" --old_tokenizer "google/flan-t5-base"
```
The new tokenizer will be saved to the newly created folder, e.g., "flan-t5-base-clinical-tokenizer". 


## Pre-train T5
To pre-train T5 model on the clinical corpus, run below command.
> **⚠ WARNING: wandb API key needed.**  
> Please put your **wandb API key** in **line 59 in run_t5_seq2seq2_lm.py** first.

```
# train command 
# single GPU
python src/run_t5_seq2seq_lm.py \
  --model_name_or_path "google/flan-t5-base" \
  --tokenizer_name "flan-t5-xxl-clinical-tokenizer" \
  --report_to wandb \
  --run_name "t5_test" \
  --train_file "Datasets/train.json" \
  --validation_file "Datasets/valid.json" \
  --streaming \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --evaluation_strategy "steps" \
  --eval_steps 5 \
  --max_steps 20 \
  --save_steps 5 \
  --predict_with_generate \
  --output_dir result \
  --overwrite_output_dir

# multiple GPU in same node (DDP)
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 \
  run_t5_seq2seq_lm.py \
  --model_name_or_path "google/flan-t5-xl" \
  --tokenizer_name "flan-t5-xxl-clinical-tokenizer" \
  --report_to wandb \
  --run_name "flan_t5_xl_csf3" \
  --train_file "Datasets/Train" \
  --streaming \
  --per_device_train_batch_size 4 \
  --do_train \
  --max_steps 400000 \
  --save_steps 30000 \
  --output_dir result \
  --overwrite_output_dir \
  --bf16 True \
  --tf32 True \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "T5Block"


# more arguments can be specified, e.g., num_train_epochs
```
- The trained model will be saved to the folder "result", including all checkpoints.
- Information logged by WanDB would be saved to the folder "wandb".

### Example of project floder
```
.
└── Project
    ├── Datasets
    │   ├── Train
    │   │   └── chunkX.json
    │   └── Validation
    ├── flan-t5-xxl-clinical-tokenizer
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    ├── result
    │   ├── checkpoint-30000
    │   │   ├── config.json
    │   │   ├── (...)
    │   │   └── training_args.bin
    │   ├── checkpoint-XXX
    │   ├── README.md
    │   ├── (...)
    │   ├── pytorch_model.bin
    │   ├── (...)
    │   └── training_args.bin
    ├── wandb
    │   └── run-[DATE]_[TIME]-[RANDOM]
    └── run_t5_seq2seq_lm.py
```
