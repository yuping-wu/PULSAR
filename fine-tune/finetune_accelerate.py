# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to train and evaluate a regular supervised model trained with Data Augumentation on Mimic-iii dataset.

Command line used in CSF3:
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file accelerate_t5_fsdp.yml 
  finetune_accelerate.py --model_path epoch_0 
    --tokenizer_name google/flan-t5-xl 
    --input_file Datasets/finetune/BioNLP2023-1A-newTrain.csv 
    --input_test_file Datasets/finetune/BioNLP2023-1A-newTest.csv
    --output_dir output --output_file system.txt 
    --predict_with_generate --num_train_epochs 1 
    --a100 --per_device_train_batch_size 4 --max_input_length 512 
    --gradient_accumulation_steps 1 --per_device_eval_batch_size 4 
    --learning_rate 3e-5 --no_val --train --dev True
"""

import json, logging
import argparse
from functools import partial
import os, math
import sys
import nltk
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import datasets
import evaluate

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_metric, Dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
import wandb

logger = get_logger(__name__)

def process_data(dg: pd.DataFrame, test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    # Create a new dataframe with only the rows of 'data' that don't match the 'Summary' column of 'test_df'
    filtered_data = dg[~dg['Summary'].isin(test_df['Summary'])]
    print("The size of the input augmented dataset is %d" %len(dg) )
    print("The size of the augmented dataset after filter is %d" %len(filtered_data))

    # Update the 'data' dataframe with the filtered data
    data = filtered_data

    # Concatenate the dataframes vertically (i.e., stack them on top of each other)
    combined_df = pd.concat([train_df, data], ignore_index=True)

    # drop rows with missing values in 'source_text' or 'target_text'
    combined_df.dropna(subset=['source_text', 'target_text'], inplace=True)

    # Reset the index of the combined dataframe
    new_train = combined_df.reset_index(drop=True)

    print("The size of the new training dataset is %d" %len(new_train))
    return new_train



def load_dataset(input_file, input_test_file, input_dg_file=None, val=True, dev=False) -> pd.DataFrame:
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(input_file)
    test_df = pd.read_csv(input_test_file)
    
    # Read the DG file only if it is not None
    if input_dg_file:
        dg = pd.read_csv(input_dg_file)
        dg['source_text'] = " <ASSESSMENT> " + dg['Assessment'] + " <SUBJECTIVE> "+ dg['Subjective Sections'] +" <OBJECTIVE> " + dg['Objective Sections']
        dg['target_text'] = dg["Summary"]
    else:
        dg = pd.DataFrame(columns=['source_text', 'Summary'])
    
    test_df['source_text'] = " <ASSESSMENT> " + test_df['Assessment'] + " <SUBJECTIVE> "+ test_df['Subjective Sections'] +" <OBJECTIVE> " + test_df['Objective Sections']
    # test_df['source_text'] = " <ASSESSMENT> " + test_df['Assessment']
    if dev:
        test_df['target_text'] = test_df["Summary"]

    # Create the source and target text columns by concatenating the other columns
    df['source_text'] = " <ASSESSMENT> " + df['Assessment'] + " <SUBJECTIVE> "+ df['Subjective Sections'] +" <OBJECTIVE> " + df['Objective Sections']
    # df['source_text'] = " <ASSESSMENT> " + df['Assessment']
    df['target_text'] = df["Summary"]

    # Convert all columns to string type
    df = df.applymap(str)
    test_df = test_df.applymap(str)

    # Split the dataframe into train, validation and test sets
    if val:
      train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2023)
    else:
      train_df = df
      valid_df = pd.DataFrame(columns=['source_text', 'Summary'])
     # remove augmented data from train_df
    if dev:
        train_df = process_data(dg, test_df, train_df)
    else:
        train_df = process_data(dg, valid_df, train_df)
      
    # Convert the pandas dataframes to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df) 
    test_dataset = Dataset.from_dict(test_df)
    
    # Create a DatasetDict object that contains the train, validation and test datasets
    my_dataset_dict = DatasetDict({"train":train_dataset,"test":test_dataset,'validation':valid_dataset})
    
    # Return the DatasetDict object
    return my_dataset_dict


def preprocess_function(examples, max_input_length, max_target_length, prefix="summarization"):
    inputs = [prefix + doc for doc in examples["source_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    # Setup the tokenizer for targets
    print(type(examples))
    try:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target_text"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
    
    except KeyError:
        pass

    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}



def save_predictions(trainer, args, tokenizer, test_results, output_file):
    """
    Saves the generated predictions to a file.
    Args:
        trainer (Trainer): The Hugging Face Trainer instance.
        args (argparse.Namespace): The command-line arguments.
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer instance.
        test_results (EvalPrediction): The evaluation predictions.
        output_file (str): The path to the output file where predictions will be saved.
    """
    if trainer.is_world_process_zero():
        if args.predict_with_generate:
            test_preds = tokenizer.batch_decode(
                test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            test_preds = [pred.strip() for pred in test_preds]
            with open(output_file, "w") as writer:
                writer.write("\n".join(test_preds))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to a csv file into model')
    parser.add_argument('--input_dg_file', type=str, default=None, 
                        help='Path to a data augmented csv file into model')
    parser.add_argument('--input_test_file', type=str, required=True, 
                        help='Path to a test csv file into model')
    parser.add_argument('--output_file', type=str, default='system.txt', 
                        help='Generate sample')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to an output directory were the finetuned model and results are saved')
    
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", 
                        help="Name of the pretrained model to use for finetuning")
    parser.add_argument("--model_path", type=str, default=None,
                    help="Path to the model directory if using a locally saved model")
    parser.add_argument("--tokenizer_name", type=str, default="google/flan-t5-base", 
                        help="Hugging Face tokenizer name")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="The training batch size per GPU")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="The eval batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="The number of gradient accumulation steps to perform")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The maximum learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="The maximum number of training epochs")
    parser.add_argument("--seeds", type=int, nargs='+', default=[2023],
                        help="The seeds to use. If multiple are given, the entire finetuning process is repeated multiple times.")
    parser.add_argument('--max_input_length', type=int, default=512, 
                        help='Maximum length of input sequence')
    parser.add_argument('--max_target_length', type=int, default=64, 
                        help='Maximum length of target sequence')
    parser.add_argument("--predict_with_generate", action="store_true", 
                        help="Whether to use generation for prediction.")
    parser.add_argument("--a100", action="store_true", 
                        help="Use BF16 and TF32.")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument('--dev', type=bool, default=False, 
                        help='Whether this run is for development or not. If for development, reference summary is required in test file.')


    args = parser.parse_args()
    val = not args.no_val
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    # some hard-coded parameters
    args.max_train_steps = None
    args.num_beams = 6
    args.with_tracking = False
    args.num_warmup_steps = 0
    args.lr_scheduler_type = "linear"
    args.resume_from_checkpoint = None


    preprocess_function = partial(preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length)

    print("model path is ", args.model_path)

    if args.model_path:
        print("Loading model from local path", args.model_path)
        logger.info('Loading model from local path ', args.model_path)
        config = AutoConfig.from_pretrained(args.model_name)
        model_path = args.model_path
        if 'pytorch_model.bin' in os.listdir(model_path):
            config = AutoConfig.from_pretrained(args.model_name)
            model = AutoModelForSeq2SeqLM.from_config(config)
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu'))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Done with loading model from local path")
        logger.info('Done with loading model from local path.')
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        print('Loading model from huggingface ', args.model_name)
        model_name = args.model_name
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # metric = load_metric("rouge")
    import evaluate
    metric = evaluate.load("rouge")
    
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = 'wnadb'
        accelerator_log_kwargs["logging_dir"] = args.output_dir
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    for seed in args.seeds:
        set_seed(seed)
        rn.seed(seed)
        output_dir = os.path.join(args.output_dir, str(seed))

        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # Load the dataset using the load_dataset function
        my_dataset_dict = load_dataset(args.input_file, args.input_test_file, args.input_dg_file, val=val, dev=args.dev)

        with accelerator.main_process_first():
            train_dataset = my_dataset_dict['train'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['train'].column_names)
            eval_dataset = my_dataset_dict['validation'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['validation'].column_names)
            test_dataset = my_dataset_dict['test'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['test'].column_names)
        
        # Log a few random samples from the training set:
        for index in rn.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8 if accelerator.use_fp16 else None)

        train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # prepare model before optimizer
        model = accelerator.prepare(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
        )

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("pretrain_t5_no_trainer", experiment_config)
        
         # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
            starting_epoch = 1

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                        accelerator.log({"train_step_loss": loss.detach().float()}, step=completed_steps)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                if completed_steps >= args.max_train_steps:
                    break
        
        if output_dir is not None:
            output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        accelerator.save_state(output_dir)

        # to make model be able to be evaluated under FSDP model
        dummy_inputs = tokenizer(
                'This is a dummy input for the purpose of FSDP wrapping',
                text_target = "OK, ignored.",
                max_length=max_input_length, truncation=True, 
                return_tensors='pt'
        )
        accelerator.wait_for_everyone()
        model.eval()

        gen_kwargs = {
            "max_length": max_target_length,
            "num_beams": args.num_beams,
            "synced_gpus": True,
        }
        # run dummy inputs
        dummy_outputs = accelerator.unwrap_model(model)(**dummy_inputs)
        predictions, references = [], []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = accelerator.gather_for_metrics(generated_tokens)

                generated_tokens = generated_tokens.cpu().numpy()
                print(generated_tokens)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                test_preds = [pred.strip() for pred in decoded_preds]
                predictions += test_preds
                
                if args.dev:
                    labels = batch['labels']
                    labels = accelerator.pad_across_processes(batch['labels'], dim=1, pad_index=tokenizer.pad_token_id)
                    labels = accelerator.gather_for_metrics(labels)
                    labels = labels.cpu().numpy()
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    test_labels = [label.strip() for label in decoded_labels]
                    references += test_labels
                    metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )
        
        if args.dev:
            result = metric.compute(use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            if accelerator.is_local_main_process:
                print("Eval result: {}".format(result))
                # save the predictions in json format
                with open(os.path.join(output_dir, "predictions.json"), "w") as fpf:
                    for p in [{"prediction" : k, "label": v} for k, v in zip(predictions, references)]: 
                        fpf.write(json.dumps(p) + "\n")

        # set the output file path
        output_test_preds_file = os.path.join(output_dir, args.output_file)
        output_test_preds_file = output_test_preds_file.replace('.txt', '.jsonl') # replace .txt with .jsonl

        if accelerator.is_local_main_process:
            with open(output_test_preds_file, "w") as writer:
                for pred in predictions:
                    json.dump(pred, writer) # write each prediction as a JSON object on a separate line
                    writer.write('\n') # add a newline character to separate each object
