# PULSAR: Pre-training with Extracted Gap Healthcare Terms for Summarizing Patients’ Problems and Data Augmentation with Black-box Large Language Models
Fine-tuning readme.

## Clef finetune
Steps to make things work (assuming you're in the root directory):

 - clone the clef repo with datasets (let's assume in `../clef-medqa`)
 - install requirements from `pre-train/requirements.txt`
 - run the following command to see if all works

```bash
 WANDB_MODE=disabled torchrun fine-tune/clef_finetune.py --model_path google/flan-t5-small --tokenizer_name google/flan-t5-small --input_file ../clef-medqa/dataset/TaskB/TaskB-TrainingSet.csv --input_val_file ../clef-medqa/dataset/TaskB/TaskB-ValidationSet.csv --output_dir ../11b-l512-lr3e-5-nodg/ --output_file system.txt --predict_with_generate --num_train_epochs 3 --per_device_train_batch_size 4 --max_input_length 512 --gradient_accumulation_steps 1 --per_device_eval_batch_size 16 --learning_rate 3e-5 --train
```

## Clef finetune with accelerate
```bash
# train (and evaluate on dev set)
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml clef_finetune_accelerate.py --model_path PATH-TO-PULSAR-ckpt --model_name google/flan-t5-xl --tokenizer_name google/flan-t5-xl --input_file PATH-TO-CLEF2023-TaskB-TrainingSet-DEV.csv --input_val_file PATH-TO-CLEF2023-TaskB-ValidationSet-DEV.csv --output_dir output_clef --max_input_length 512 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 2  --learning_rate 3e-5 --train


# test (load trained model and predict on test set)
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml clef_finetune_accelerate.py --model_path output_clef/2023/epoch_1 --model_name google/flan-t5-xl --tokenizer_name google/flan-t5-xl --input_file PATH-TO-CLEF2023-TaskB-TrainingSet-DEV.csv --input_test_file PATH-TO-CLEF2023-TaskB-ValidationSet-DEV.csv --output_dir output_clef --output_file system.txt --max_input_length 512 --per_device_eval_batch_size 4  --test

```


### TODO
- Adapt the accelerate code. (Done)
- Evaluate bigger models.
- etc
