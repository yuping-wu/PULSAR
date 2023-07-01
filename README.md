# PULSAR: Pre-training with Extracted Gap Healthcare Terms for Summarizing Patients’ Problems and Data Augmentation with Black-box Large Language Models
Code repository for PULASR, including
1. Pre-training T5 with gap text spans generation as the pre-training objective
2. Fine-tuning the pre-trained model with data agumentation on the downstream task, i.e., [BioNLP Workshop 2023 Shared Task 1A: Problem List Summarization](https://physionet.org/content/bionlp-workshop-2023-task-1a/1.1.0/)

## Pre-training
See `pre-train/README.md` for details.

## Fine-tuning
See `fine-tune/README.md` for details.

## 📕 Citation

If you make use of the code in this repository or of any follow up works, please cite the following paper(s):
````
@article{li2023pulsar,
  title={PULSAR: Pre-training with Extracted Healthcare Terms for Summarising Patients' Problems and Data Augmentation with Black-box Large Language Models},
  author={Li, Hao and Wu, Yuping and Schlegel, Viktor and Batista-Navarro, Riza and Nguyen, Thanh-Tung and Kashyap, Abhinav Ramesh and Zeng, Xiaojun and Beck, Daniel and Winkler, Stefan and Nenadic, Goran},
  journal={arXiv preprint arXiv:2306.02754},
  year={2023}
}
````
