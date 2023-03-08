# Codes for pre-train T5 on clinical corpus
Repository for
1. Train a new tokenizer on clinical corpus from the old T5 tokenizer
2. Pre-train T5 model on clinical corpus with Gap Span/Sentence Generation as the pre-training objective


## Requirements
- python==3.9.16
- transformers==4.26.1
- pytorch==1.13.1
- sentencepiece==0.1.97
- nltk==3.8.1
- datasets==2.9.0
- evaluate==0.4.0
- scikit-learn==1.2.1


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
```
# train command
python src/run_t5_seq2seq_lm.py \
    --model_name_or_path "google/flan-t5-base" \
    --tokenizer_name "flan-t5-base-clinical-tokenizer" \
    --train_file "Datasets/train_example.json" \
    --do_train \
    --output_dir result \
    --overwrite_output_dir

# more arguments can be specified, e.g., num_train_epochs
```
The trained model will be saved to the folder "result".