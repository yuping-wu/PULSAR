# re-save model

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint_path = 'output/epoch_2/pytorch_model.bin'
model_name = 'google/flan-t5-xxl'

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_config(config)
print('Done with loading model from config')
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
print('Done with loading model state dict')

# save model and tokenizer
tokenizer.save_pretrained('output/epoch_2_saved/')
model.save_pretrained('output/epoch_2_saved/')
