# Codes for pre-train T5 on clinical corpus

## Requirements
- python 3.9.16
- pytorch 1.13.1
- CUDA 11.6


## Pre-train T5

> **⚠ WARNING: wandb API key needed.**  
> If using WandDB to log experiments, ***wandb.login()*** should be run first to store the valid API key.

To pre-train T5 model on the clinical corpus, using acclerate library, run below command.

```
# train command
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file accelerate_t5_fsdp.yml accelerate_t5.py config.json
```

- ***accelerate_t5_fsdp.yml***: configuration file for the library accelerate to specify number of GPUs, whether to use FSDP, etc.

- ***config.json***: configuration file for the script accelerate_t5.py. Explanation about each parameter in the file can be found in the script.

- The trained model will be saved to the folder "result", including all checkpoints.
- Information logged by WanDB would be saved to the folder "wandb".

### Example of project floder
```
.
└── Project
    ├── Datasets
    │   ├── Train
    │   │   └── chunkX.json
    │   └── Validation
    ├── result
    │   ├── step_3000
    │   │   ├── optimizer.bin
    │   │   ├── pytorch_model.bin
    │   │   ├── random_states_0.pkl
    │   │   ├── random_states_1.pkl
    │   │   └── scheduler.bin
    │   ├── (...)
    │   ├── all_results.json
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── pytorch_model.bin
    │   └── (...)
    ├── wandb
    │   └── run-[DATE]_[TIME]-[RANDOM]
    ├── accelerate_t5.py
    ├── accelerate_t5_fsdp.yml
    └── config.json
```
