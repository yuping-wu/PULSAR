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
This script is used to finetune the PULSAR model on CLEF-MedQA Task2 dataset.
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
from tqdm.auto import tqdm

import datasets
import evaluate

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict

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
    filtered_data = dg[~dg["Summary"].isin(test_df["Summary"])]
    print("The size of the input augmented dataset is %d" % len(dg))
    print("The size of the augmented dataset after filter is %d" % len(filtered_data))

    # Update the 'data' dataframe with the filtered data
    data = filtered_data

    # Concatenate the dataframes vertically (i.e., stack them on top of each other)
    combined_df = pd.concat([train_df, data], ignore_index=True)

    # drop rows with missing values in 'source_text' or 'target_text'
    combined_df.dropna(subset=["source_text", "target_text"], inplace=True)

    # Reset the index of the combined dataframe
    new_train = combined_df.reset_index(drop=True)

    print("The size of the new training dataset is %d" % len(new_train))
    return new_train

HEADERS = {'PLAN': "Conversation about the patient's treatment plan:",
 'ASSESSMENT': "Conversation about the patient's medical assessment:",
 'ALLERGY': "Conversation about the patient's allergies:",
 'CC': "Conversation about the patient's chief complaint:",
 'ROS': "Conversation about the review of patient's systems:",
 'FAM/SOCHX': "Conversation about the patient's social and family history:",
 'PASTMEDICALHX': "Conversation about the patient's past medical history:",
 'DIAGNOSIS': "Conversation about the patient's diagnosis:",
 'DISPOSITION': "Conversation about the patient's disposition:",
 'GENHX': "Conversation about the patient's history of present illness",
 'IMAGING': "Conversation about the patient's imaging results",
 'LABS': "Conversation about the patient's lab results:",
 'MEDICATIONS': "Conversation about the patient's current medications:",
 'PASTSURGICAL': "Conversation about the patient's past surgical history:",
 'EXAM': "Conversation about the patient's examination results:",
 'PROCEDURES': "Conversation about the procedures performed on the patient:",
 'IMMUNIZATIONS': "Conversation about the patient's vaccinations:",
 'OTHER_HISTORY': "Conversation about the patient's other history:",
 'GYNHX': "Conversation about the patient's gynecologic history:"}

def load_dataset(
    input_file, input_val_file, input_test_file, input_dg_file=None, val=True, header_input=False, convert_header=False
) -> pd.DataFrame:
    # Load the CSV file into a pandas dataframe
    train_df = pd.read_csv(input_file)

    print(train_df.columns)
    # Read the DG file only if it is not None
    if input_dg_file:
        dg = pd.read_csv(input_dg_file)
        if convert_header:
            dg['section_header'] = dg['section_header'].apply(lambda x: HEADERS[x])
        dg["source_text"] = (dg["section_header"] + "\n" if header_input else "") + dg["dialogue"]
        dg["target_text"] = (dg["section_header"] + "\n" if not header_input else "") + dg["section_text"]
    else:
        dg = pd.DataFrame(columns=["source_text", "Summary"])

    if input_test_file:
        test_df = pd.read_csv(input_test_file)
        if convert_header:
            test_df['section_header'] = test_df['section_header'].apply(lambda x: HEADERS[x])
        test_df["source_text"] = (test_df["section_header"] + "\n" if header_input else "") + test_df["section_text"]
    else:
        test_df = pd.DataFrame(columns=["source_text", "Summary"])

    if input_val_file:
        val_df = pd.read_csv(input_val_file)
        if convert_header:
            val_df['section_header'] = val_df['section_header'].apply(lambda x: HEADERS[x])
        val_df["source_text"] = (val_df["section_header"] + "\n" if header_input else "") + val_df["dialogue"]
        val_df["target_text"] = (val_df["section_header"] + "\n" if not header_input else "") + val_df["section_text"]
    else:
        val_df = pd.DataFrame(columns=["source_text", "Summary"])
    # Create the source and target text columns by concatenating the other columns
    if convert_header:
            train_df['section_header'] = train_df['section_header'].apply(lambda x: HEADERS[x])
    train_df["source_text"] = (train_df["section_header"] + "\n" if header_input else "") + train_df["dialogue"]
    train_df["target_text"] = (train_df["section_header"] + "\n" if not header_input else "") + train_df["section_text"]

    # Convert all columns to string type
    train_df = train_df.applymap(str)
    test_df = test_df.applymap(str)
    val_df = val_df.applymap(str)
    print(train_df.head())
    print(val_df.head())
    # Split the dataframe into train, validation and test sets
    # Concatenate the dataframes vertically (i.e., stack them on top of each other)
    if val:
        combined_df = pd.concat([train_df, dg], ignore_index=True)
    else:
        combined_df = pd.concat([train_df, val_df, dg], ignore_index=True)

    # drop rows with missing values in 'source_text' or 'target_text'
    combined_df.dropna(subset=["source_text", "target_text"], inplace=True)

    # Reset the index of the combined dataframe
    combined_df = combined_df.reset_index(drop=True)

    print("The size of the new training dataset is %d" % len(combined_df))

    # Convert the pandas dataframes to Hugging Face datasets
    train_dataset = Dataset.from_pandas(combined_df)
    valid_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_dict(test_df)

    # Create a DatasetDict object that contains the train, validation and test datasets
    my_dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": valid_dataset})

    # Return the DatasetDict object
    return my_dataset_dict



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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True, help="Path to a csv file into model")
    parser.add_argument("--input_dg_file", type=str, default=None, help="Path to a data augmented csv file into model")
    parser.add_argument("--input_test_file", type=str, required=False, help="Path to a test csv file into model")
    parser.add_argument("--input_val_file", type=str, required=False, help="Path to a val csv file into model")
    parser.add_argument("--output_file", type=str, default="system.txt", help="Generate sample")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to an output directory were the finetuned model and results are saved",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the pretrained model to use for finetuning",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the model directory if using a locally saved model"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="google/flan-t5-base", help="Hugging Face tokenizer name"
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="The training batch size per GPU")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="The eval batch size per GPU")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps to perform",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The maximum learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The maximum number of training epochs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2023],
        help="The seeds to use. If multiple are given, the entire finetuning process is repeated multiple times.",
    )
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum length of input sequence")
    parser.add_argument("--max_target_length", type=int, default=64, help="Maximum length of target sequence")
    parser.add_argument(
        "--predict_with_generate", action="store_true", help="Whether to use generation for prediction."
    )
    parser.add_argument("--a100", action="store_true", help="Use BF16 and TF32.")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test mode")    
    parser.add_argument("--header_input", action="store_true", help='whether to use the section header as part of input.')
    parser.add_argument("--header_output", action="store_true", help='whether to use the section header as part of output.')
    parser.add_argument("--convert_header", action="store_true", help='whether to convert the section header into natural language.')
    parser.add_argument("--prefix", 
                        type=str, 
                        default="summarize: ", 
                        help="prefix for flan-T5")

    args = parser.parse_args()
    val = not args.no_val
    assert args.input_test_file or args.input_val_file, "Do you want to train only? Really?"
    args.input_test_file = args.input_test_file or args.input_val_file
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    prefix = args.prefix

    # some hard-coded parameters
    args.max_train_steps = None
    args.num_beams = 6
    args.with_tracking = False
    args.num_warmup_steps = 0
    args.lr_scheduler_type = "linear"
    args.resume_from_checkpoint = None

    # preprocess_function = partial(
    #     preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length
    # )

    print('model_path is ', args.model_path)
    if args.model_path:
        # print('Loading model from local path ', args.model_path)
        logger.info('Loading model from local path ', args.model_path)
        model_path = args.model_path
        if 'pytorch_model.bin' in os.listdir(model_path) and args.model_name is not None:
            config = AutoConfig.from_pretrained(args.model_name)
            model = AutoModelForSeq2SeqLM.from_config(config)
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu'))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # print('Done with loading model from local path.')
        logger.info('Done with loading model from local path.')
    else:
        model_name = args.model_name
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Metric
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

        def preprocess_function(examples):
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

        # input_file, input_val_file, input_test_file, input_dg_file=None, val=True
        my_dataset_dict = load_dataset(
            input_file=args.input_file,
            input_val_file=args.input_val_file,
            input_test_file=args.input_test_file,
            input_dg_file=args.input_dg_file,
            val=val,
            header_input=args.header_input,
            header_output=args.header_output,
            convert_header=args.convert_header
        )
        with accelerator.main_process_first():
            train_dataset = my_dataset_dict['train'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['train'].column_names, load_from_cache_file=False)
            eval_dataset = my_dataset_dict['validation'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['validation'].column_names, load_from_cache_file=False)
            test_dataset = my_dataset_dict['test'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['test'].column_names, load_from_cache_file=False)
        
        # Log a few random samples from the training set:
        if args.train:
            for index in rn.sample(range(len(train_dataset)), 1):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        if args.test:
            for index in rn.sample(range(len(test_dataset)), 1):
                logger.info(f"Sample {index} of the testing set: {test_dataset[index]}.")
        

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8 if accelerator.use_fp16 else None)

        train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # prepare model before optimizer
        model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                model, train_dataloader, eval_dataloader, test_dataloader
            )
        
        # to make model be able to be evaluated if wrapped by FSDP mode
        dummy_inputs = tokenizer(
            'This is a dummy input for the purpose of FSDP wrapping',
            text_target = "OK, ignored.",
            max_length=args.max_input_length, padding=False, truncation=True, 
            return_tensors='pt'
        )

        if args.train:
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

            optimizer, lr_scheduler = accelerator.prepare(
                optimizer, lr_scheduler
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
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                    dirs.sort(key=os.path.getctime)
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                else:
                    resume_step = int(training_difference.replace("step_", ""))
                    starting_epoch = resume_step // len(train_dataloader)
                    resume_step -= starting_epoch * len(train_dataloader)
            
            for epoch in range(starting_epoch, args.num_train_epochs):
                model.train()
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            completed_steps += 1
                            continue

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

                
                # save training state for each epoch
                # accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))
                    
            # save final-epoch model
            accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))

            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(
            #     os.path.join(output_dir, f"epoch_{epoch}"), is_main_process=accelerator.is_main_process, 
            #     save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
            # )
              
            # do validation
            if val:
                model.eval()
                gen_kwargs = {
                    "max_length": max_target_length,
                    "num_beams": args.num_beams,
                    "synced_gpus": True,
                }
                # run dummy inputs
                _ = accelerator.unwrap_model(model)(**dummy_inputs)
                epoch_predictions, epoch_labels = [], []
                for e_step, e_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            e_batch["input_ids"],
                            attention_mask=e_batch["attention_mask"],
                            **gen_kwargs,
                        )
                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = e_batch["labels"]
                        labels = accelerator.pad_across_processes(e_batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                        generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                        generated_tokens = generated_tokens.cpu().numpy()
                        labels = labels.cpu().numpy()
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                            
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        
                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                        epoch_predictions += decoded_preds
                        epoch_labels += decoded_labels
                        metric.add_batch(
                            predictions=decoded_preds,
                            references=decoded_labels,
                        )
                
                result = metric.compute(use_stemmer=True)
                result = {k: round(v * 100, 4) for k, v in result.items()}

                logger.info(f"Eval results at epoch {epoch}: {result}")

                # save the predictions
                with open(os.path.join(output_dir, f"predictions_{epoch}.json"), "w") as pf:
                    for p in [{"prediction" : k, "label": v} for k, v in zip(epoch_predictions, epoch_labels)]:
                        pf.write(json.dumps(p) + "\n")
                

        # do test
        if args.test:
            logger.info("***** Running testing *****")
            logger.info(f"  Num examples = {len(test_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
            model.eval()
            model.config.use_cache = True
            gen_kwargs = {
                "max_length": max_target_length,
                "num_beams": args.num_beams,
                "synced_gpus": True,
            }

             # run dummy inputs
            _ = accelerator.unwrap_model(model)(**dummy_inputs)
            test_predictions = []
            test_labels = []
            for t_step, t_batch in enumerate(test_dataloader):
                generated_tokens = accelerator.unwrap_model(model).generate(
                    t_batch["input_ids"],
                    attention_mask=t_batch["attention_mask"],
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
                decoded_preds = [pred.strip() for pred in decoded_preds]
                test_predictions += decoded_preds

            # save the predictions
            output_test_preds_file = os.path.join(output_dir, args.output_file)
            output_test_preds_file = output_test_preds_file.replace('.txt', '.jsonl') # replace .txt with .jsonl

            with open(output_test_preds_file, "w") as writer:
                for pred in test_predictions:
                    json.dump(pred, writer) # write each prediction as a JSON object on a separate line
                    writer.write('\n') # add a newline character to separate each object