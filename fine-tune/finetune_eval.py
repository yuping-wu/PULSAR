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
"""

# command line to run on csf3
# CUDA_LAUNCH_BLOCKING=1 python finetune_eval.py --model_name google/flan-t5-xxl --tokenizer_name google/flan-t5-xxl --input_file Datasets/finetune/BioNLP2023-1A-Train.csv --input_test_file Datasets/finetune/BioNLP2023-1A-Test.csv --output_dir output/2023/epoch_0 --output_file system.txt --max_input_length 512 --per_device_eval_batch_size 2

import json
import argparse
from functools import partial
import os
import sys
import nltk
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import T5Tokenizer, T5Model,T5ForConditionalGeneration,T5Config, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup,PegasusConfig, PegasusTokenizer, PegasusForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch


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



def load_dataset(input_file, input_test_file, input_dg_file=None, val=True) -> pd.DataFrame:
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

    # Create the source and target text columns by concatenating the other columns
    df['source_text'] = " <ASSESSMENT> " + df['Assessment'] + " <SUBJECTIVE> "+ df['Subjective Sections'] +" <OBJECTIVE> " + df['Objective Sections']
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
    

    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="The eval batch size per GPU")
    parser.add_argument('--max_input_length', type=int, default=512, 
                        help='Maximum length of input sequence')
    parser.add_argument('--max_target_length', type=int, default=64, 
                        help='Maximum length of target sequence')


    args = parser.parse_args()
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    output_dir = args.output_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # accelerator = Accelerator()

    preprocess_function = partial(preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length)

    # config = AutoConfig.from_pretrained(args.model_path)
    # with init_empty_weights():
    #     model = AutoModelForSeq2SeqLM.from_config(config)
    # model.tie_weights()
    # model = load_checkpoint_and_dispatch(
    #     model, args.model_path, device_map="auto"
    #     )
    # model = model.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location='cpu'))
    model.to(device)
    print('model device is', model.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


    # Load the dataset using the load_dataset function
    my_dataset_dict = load_dataset(args.input_file, args.input_test_file, args.input_dg_file, val=False)
    eval_dataset = my_dataset_dict['test'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['test'].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    print("Evaluating on Test set")

    model.eval()
    print('model device is ', model.device)

    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": 6,
    }

    # dummy_inputs = tokenizer(
    #         'This is a dummy input for the purpose of FSDP wrapping',
    #         text_target = "OK, ignored.",
    #         max_length=512, truncation=True, 
    #         return_tensors='pt'
    # )
    # trainer.model.to('cuda')
    predictions = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            print(generated_tokens)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            test_preds = [pred.strip() for pred in decoded_preds]
            predictions += test_preds
    
    # set the output file path
    output_test_preds_file = os.path.join(output_dir, args.output_file)
    output_test_preds_file = output_test_preds_file.replace('.txt', '.jsonl') # replace .txt with .jsonl

    with open(output_test_preds_file, "w") as writer:
        for pred in predictions:
            json.dump(pred, writer) # write each prediction as a JSON object on a separate line
            writer.write('\n') # add a newline character to separate each object


    # save_predictions(trainer, args, tokenizer, test_result, output_file)
