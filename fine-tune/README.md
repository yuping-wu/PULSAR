# Codes for fine-tuning models on downstream summarization tasks

## Requirements
- install requirements from `pre-train/requirements.txt`

## BioNLP finetune
```bash
CUDA_LAUNCH_BLOCKING=1 python finetune_trainer.py --model_path google/flan-t5-small --tokenizer_name google/flan-t5-small --input_file PATH/TO/BioNLP2023-1A-newTrain.csv --input_dg_file PATH/TO/dg_1K_train.csv --input_test_file PATH/TO/BioNLP2023-1A-newTest.csv --output_dir output --predict_with_generate --num_train_epochs 15 --per_device_train_batch_size 32 --max_input_length 512 --gradient_accumulation_steps 1 --per_device_eval_batch_size 32 --learning_rate 2e-5 --no_val --train
```

## BioNLP finetune with accelerate (distributed training for large models)
```bash
# train and test
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml finetune_accelerate.py --model_name PATH/TO/PULSAR-ckpt --tokenizer_name PATH/TO/PULSAR/CKPT --input_file PATH/TO/BioNLP2023-1A-Train.csv --input_test_file PATH/TO/BioNLP2023-1A-DEV-Test.csv --output_dir PATH/TO/OUTPUT --output_file system.txt --max_input_length 512 --per_device_eval_batch_size 2 --per_device_train_batch_size 4 --num_train_epochs 2 --train
```

## Clef finetune
Steps to make things work (assuming you're in the root directory):

 - clone the clef repo with datasets (let's assume in `../clef-medqa`)
 - run the following command to see if all works

```bash
 WANDB_MODE=disabled torchrun fine-tune/clef_finetune.py --model_path google/flan-t5-small --tokenizer_name google/flan-t5-small --input_file ../clef-medqa/dataset/TaskB/TaskB-TrainingSet.csv --input_val_file ../clef-medqa/dataset/TaskB/TaskB-ValidationSet.csv --output_dir ../11b-l512-lr3e-5-nodg/ --output_file system.txt --predict_with_generate --num_train_epochs 3 --per_device_train_batch_size 4 --max_input_length 512 --gradient_accumulation_steps 1 --per_device_eval_batch_size 16 --learning_rate 3e-5 --train
```

## Clef finetune with accelerate (distributed training for large models)
### Task B
```bash
# train (and evaluate on dev set)
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml clef_finetune_accelerate.py --model_path PATH/TO/PULSAR-ckpt --model_name google/flan-t5-xl --tokenizer_name google/flan-t5-xl --input_file PATH/TO/CLEF2023-TaskB-TrainingSet.csv --input_val_file PATH-TO-CLEF2023-TaskB-ValidationSet.csv --output_dir output_clef --max_input_length 512 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 2  --learning_rate 3e-5 --train


# test (load trained model and predict on test set)
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml clef_finetune_accelerate.py --model_path output_clef/2023/epoch_1 --model_name google/flan-t5-xl --tokenizer_name google/flan-t5-xl --input_file PATH-TO-CLEF2023-TaskB-TrainingSet.csv --input_test_file PATH-TO-CLEF2023-TaskB-TestSet.csv --output_dir output_clef --output_file system.txt --max_input_length 512 --per_device_eval_batch_size 4  --test
```

### Task C
```bash
# train and test
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml clef_finetune_accelerate_taskc.py --model_path PATH-TO-PULSAR-ckpt --model_name google/flan-t5-xl --tokenizer_name google/flan-t5-xl --input_file PATH-TO-CLEF2023-TaskC-TrainingSet.csv --input_val_file PATH-TO-CLEF2023-TaskB-TestSet.csv --output_dir output_clef --max_input_length 512 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 2  --learning_rate 3e-5 --train --test
```

## Explanation for other useful files
- `convert_statedict.py`: convert the saved state dict from accelerate to the one from_pretrained() can load
- `finetune_eval.py`: evaluate the fine-tuned model on the test set only
- `inference_llama.py`: inference on the trained LLAMA (with LoRA) model
- `inference_t5.py`: inference on the trained PULSAR-series