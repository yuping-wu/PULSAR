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



import argparse
from functools import partial
import os
import nltk
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import T5Tokenizer, T5Model,T5ForConditionalGeneration,T5Config
from transformers import AdamW, get_linear_schedule_with_warmup,PegasusConfig, PegasusTokenizer, PegasusForConditionalGeneration


def load_dataset(input_file, input_test_file):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(input_file)
    test_df = pd.read_csv(input_test_file)

    # Create the source and target text columns by concatenating the other columns
    df['source_text'] = " <ASSESSMENT> " + df['Assessment'] + " <SUBJECTIVE> "+ df['Subjective Sections'] +" <OBJECTIVE> " + df['Objective Sections']
    df['target_text'] = df["Summary"]

    test_df['source_text'] = " <ASSESSMENT> " + test_df['Assessment'] + " <SUBJECTIVE> "+ test_df['Subjective Sections'] +" <OBJECTIVE> " + test_df['Objective Sections']
    test_df['target_text'] = test_df["Summary"]

    # Convert all columns to string type
    df = df.applymap(str)
    test_df = test_df.applymap(str)

    # Split the dataframe into train, validation and test sets
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2023)
    #test_df, valid_df = train_test_split(test_df, test_size=0.2, random_state=2023)
    

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
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
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


    args = parser.parse_args()

    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    preprocess_function = partial(preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length)

    if args.model_path:
        model_path = args.model_path
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        model_name = args.model_name
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    metric = load_metric("rouge")

    for seed in args.seeds:
        output_dir = os.path.join(args.output_dir, str(seed))

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir, num_train_epochs=args.num_train_epochs, 
            per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate, weight_decay=0.01, evaluation_strategy="epoch",
            seed=seed, save_total_limit=3, predict_with_generate=True,
            fp16=False, push_to_hub=False,
        )

        # Load the dataset using the load_dataset function
        my_dataset_dict = load_dataset(args.input_file, args.input_test_file)

        tokenized_datasets = my_dataset_dict.map(preprocess_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


        trainer = Seq2SeqTrainer(
            model=model, 
            args=training_args, 
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        
        print("Evaluating on Test set")
        test_result = trainer.predict(
            tokenized_datasets["test"],
            metric_key_prefix="test",
            max_length=max_target_length,
            num_beams=6
        )
        print(test_result)
        print("Writing to the file")


        if trainer.is_world_process_zero():
            if args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_result.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]

                # set the output file path
                output_test_preds_file = os.path.join(output_dir, args.output_file)
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

        # save_predictions(trainer, args, tokenizer, test_result, output_file)
