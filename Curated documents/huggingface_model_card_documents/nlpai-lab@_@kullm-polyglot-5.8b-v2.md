---
language:
- ko
license: apache-2.0
datasets:
- nlpai-lab/kullm-v2
---

# KULLM-Polyglot-5.8B-v2

This model is a parameter-efficient fine-tuned version of [EleutherAI/polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b) on a KULLM v2 

Detail Codes are available at [KULLM Github Repository](https://github.com/nlpai-lab/KULLM)


## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 3e-4
- train_batch_size: 128
- seed: 42
- distributed_type: multi-GPU (A100 80G)
- num_devices: 4
- gradient_accumulation_steps: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 8.0

### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3