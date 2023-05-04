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


import json
import argparse
from functools import partial
import os
import sys
import nltk
import numpy as np
import pandas as pd
import random as rn
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    PegasusConfig,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)


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


def load_dataset(
    input_file, input_val_file, input_test_file, input_dg_file=None, val=True, header_input=False
) -> pd.DataFrame:
    # Load the CSV file into a pandas dataframe
    train_df = pd.read_csv(input_file)

    print(train_df.columns)
    # Read the DG file only if it is not None
    if input_dg_file:
        dg = pd.read_csv(input_dg_file)
        dg["source_text"] = (dg["section_header"] + "\n" if header_input else "") + dg["dialogue"]
        dg["target_text"] = (dg["section_header"] + "\n" if not header_input else "") + dg["section_text"]
    else:
        dg = pd.DataFrame(columns=["source_text", "Summary"])

    if input_test_file:
        test_df = pd.read_csv(input_test_file)
        test_df["source_text"] = (test_df["section_header"] + "\n" if header_input else "") + test_df["section_text"]
    else:
        test_df = pd.DataFrame(columns=["source_text", "Summary"])

    if input_val_file:
        val_df = pd.read_csv(input_val_file)
        val_df["source_text"] = (val_df["section_header"] + "\n" if header_input else "") + val_df["dialogue"]
        val_df["target_text"] = (val_df["section_header"] + "\n" if not header_input else "") + val_df["section_text"]
    else:
        val_df = pd.DataFrame(columns=["source_text", "Summary"])
    # Create the source and target text columns by concatenating the other columns
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


def preprocess_function(examples, max_input_length, max_target_length, prefix="summarize: "):
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
        default="google/flan-t5-base",
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
    parser.add_argument("--header_input", action="store_true", help='whether to use the section header as part of input.')

    args = parser.parse_args()
    val = not args.no_val
    assert args.input_test_file or args.input_val_file, "Do you want to train only? Really?"
    args.input_test_file = args.input_test_file or args.input_val_file
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    preprocess_function = partial(
        preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length
    )

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
            output_dir=output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.0,
            evaluation_strategy="epoch" if val else "no",
            logging_strategy="epoch",
            seed=seed,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=False,
            push_to_hub=False,
            bf16=args.a100,
            tf32=args.a100,
            gradient_checkpointing=True,
            save_strategy="no",
            # fsdp='full_shard auto_wrap offload',fsdp_transformer_layer_cls_to_wrap='T5Block'
        )

        # Load the dataset using the load_dataset function
        # input_file, input_val_file, input_test_file, input_dg_file=None, val=True
        my_dataset_dict = load_dataset(
            input_file=args.input_file,
            input_val_file=args.input_val_file,
            input_test_file=args.input_test_file,
            input_dg_file=args.input_dg_file,
            val=val,
            header_input=args.header_input,
        )

        tokenized_datasets = my_dataset_dict.map(preprocess_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        model.config.use_cache = False
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        if args.train:
            trainer.train()

        print("Evaluating on Test set")
        # from accelerate import Accelerator
        # accelerator = Accelerator()
        # trainer.model = accelerator.unwrap_model(trainer.model)
        # trainer.model.save_pretrained(output_dir)
        # to make model be able to be evaluated under FSDP model
        trainer.model.config.use_cache = True
        # dummy_inputs = tokenizer(
        #         'This is a dummy input for the purpose of FSDP wrapping',
        #         text_target = "OK, ignored.",
        #         max_length=512, truncation=True,
        #         return_tensors='pt'
        # )
        # _ = trainer.model(**dummy_inputs)

        model.config.use_cache = True
        if args.input_test_file:
            test_result = trainer.predict(
                tokenized_datasets["test"],
                metric_key_prefix="test",
                max_length=max_target_length,
                num_beams=6,
                use_cache=True,
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
                    output_test_preds_file = output_test_preds_file.replace(
                        ".txt", ".jsonl"
                    )  # replace .txt with .jsonl

                    with open(output_test_preds_file, "w") as writer:
                        for pred in test_preds:
                            json.dump(pred, writer)  # write each prediction as a JSON object on a separate line
                            writer.write("\n")  # add a newline character to separate each object

        # save_predictions(trainer, args, tokenizer, test_result, output_file)
