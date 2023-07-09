# PULSAR: Pre-training with Extracted Gap Healthcare Terms for Summarizing Patients’ Problems and Data Augmentation with Black-box Large Language Models
Code repository for PULASR, including
1. Pre-training T5 with the generation of the gap healthcare text spans as the pre-training objective
2. Fine-tuning the pre-trained model with data augmentation on the downstream summarization tasks, i.e., [BioNLP Workshop 2023 Shared Task 1A: Problem List Summarization](https://physionet.org/content/bionlp-workshop-2023-task-1a/1.1.0/) and [ImageCLEF MEDIQA-Sum-2023 Subtask B and C](https://www.imageclef.org/2023/medical/mediqa)

## Paper Link: 
- [PULSAR: Pre-training with Extracted Gap Healthcare Terms for Summarizing Patients’ Problems and Data Augmentation with Black-box Large Language Models](https://arxiv.org/abs/2306.02754)
- Large Language Models Augmented by Synthetic Dialogue Convert Patient Dialogues to Medical Records

## Pre-training
See `pre-train/README.md` for details.

## Data Augmentation for BioNLP task
See `data-augmentation/README.md` for details.

## Data Augmentation for CLEF tasks
See `aug-openai/README.md` for details.

## Fine-tuning
See `fine-tune/README.md` for details.

## Fine-tuning LLAMA with LoRA for CLEF tasks
See `stanford_alpaca/README.md` for details, e.g., requirements.
```bash
# train
CUDA_LAUNCH_BLOCKING=1 python stanford_alpaca/train.py --model_name_or_path PATH/TO/llama/13B_convert --data_path PATH/TO/CLEF_TaskB/trainset.jsonl --bf16 True --output_dir output_llama --num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 3e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True

# inference
CUDA_LAUNCH_BLOCKING=1 python fine-tune/inference_llama.py --base_model PATH/TO/llama/13B_convert --adapter_path output_llama --test_dataset PATH/TO/CLEF_TaskB/taskB_testset4participants_inputHeadersAndConversations.csv --is_causal --load_in_8bit --max_new_tokens 240
```

## Acknowledgements
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [DINO](https://github.com/timoschick/dino)

## 📕 Citation

If you make use of the code in this repository or of any follow up works, please cite the following paper(s):
````
@article{li2023pulsar,
  title={PULSAR: Pre-training with Extracted Healthcare Terms for Summarising Patients' Problems and Data Augmentation with Black-box Large Language Models},
  author={Li, Hao and Wu, Yuping and Schlegel, Viktor and Batista-Navarro, Riza and Nguyen, Thanh-Tung and Kashyap, Abhinav Ramesh and Zeng, Xiaojun and Beck, Daniel and Winkler, Stefan and Nenadic, Goran},
  journal={arXiv preprint arXiv:2306.02754},
  year={2023}
}

@article{schlegel2023pulsar,
  title={PULSAR at MEDIQA-Sum 2023: Large Language Models Augmented by Synthetic Dialogue Convert Patient Dialogues to Medical Records},
  author={Schlegel, Viktor and Li, Hao and Wu, Yuping and Subramanian, Anand and Nguyen, Thanh-Tung and Kashyap, Abhinav Ramesh and Beck, Daniel and Zeng, Xiaojun and Batista-Navarro, Riza Theresa and Winkler, Stefan and others},
  journal={arXiv preprint arXiv:2307.02006},
  year={2023}
}
````
