import os
import fire
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


def infer(
    model_path = "PULSAR-11B", 
    test_dataset = "Datasets/clef_finetune/TaskB/taskB_testset4participants_inputHeadersAndConversations.csv",
    output_file = "output_clef/system_predictions.csv",
    prefix = "summairze: ",
    id_column: str = 'ID',
    note_column: str = 'prediction',
    input_column: str = 'dialogue'
    ):
    
    print(f"Loading model from {model_path}.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")

    ds = pd.read_csv(test_dataset).to_dict(orient='records')
    results = []
    model.eval()
    for i, example in enumerate(tqdm(ds)):
        i = example.get(id_column, None) or i
        input_text = prefix + example['dialogue']
        input_ids = tokenizer(input_text, max_length=512, padding=False, truncation=True, return_tensors="pt").input_ids
        output = model.generate(input_ids.cuda(), max_length=214, num_beams=6)
        prediction = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        results.append({id_column: i, note_column: prediction, input_column: example['dialogue']})
    pd.DataFrame(results).to_csv(output_file, index=False)

if __name__ == "__main__":
    fire.Fire(infer)
