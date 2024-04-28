---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- f1
base_model: bert-base-uncased
model-index:
- name: fundhub_category_v1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# fundhub_category_v1

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1340
- F1: 0.9626

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step  | Validation Loss | F1     |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| 0.1648        | 1.0   | 25275 | 0.1583          | 0.9512 |
| 0.1396        | 2.0   | 50550 | 0.1578          | 0.9583 |
| 0.0934        | 3.0   | 75825 | 0.1340          | 0.9626 |


### Framework versions

- Transformers 4.34.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.4
- Tokenizers 0.14.1
