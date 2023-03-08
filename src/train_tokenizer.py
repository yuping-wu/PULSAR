'''
This script is for training a new toknizer based on an old one.
Corpus path must be provided.
Codes are copied from https://huggingface.co/course/chapter6/2
'''

import argparse, os
from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default='.', required=True, help="Path to the pre-training corpus.")
    parser.add_argument("--old_tokenizer", type=str, default="google/flan-t5-base", help="Name of the starting tokenizer in Huggingface.")
    args = parser.parse_args()

    # load corpus
    raw_datasets = load_dataset(
            'json',
            data_files=args.corpus_path,
        )
    
    # generate generator
    training_corpus = (raw_datasets['train'][i: i+1000]['text']
                       for i in range(0, len(raw_datasets['train']), 1000))
    
    # train new tokenizer from the old one
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer)
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, old_tokenizer.vocab_size)

    # save tokenizer
    tokenizer_name = args.old_tokenizer.split('/')[-1] + '-clinical-tokenizer'
    if not os.path.exists(tokenizer_name):
        os.makedirs(tokenizer_name)
    tokenizer.save_pretrained(tokenizer_name)
