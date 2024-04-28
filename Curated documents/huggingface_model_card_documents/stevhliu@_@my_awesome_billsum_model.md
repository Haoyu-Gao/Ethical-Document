---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- billsum
metrics:
- rouge
model-index:
- name: my_awesome_billsum_model
  results:
  - task:
      type: text2text-generation
      name: Sequence-to-sequence Language Modeling
    dataset:
      name: billsum
      type: billsum
      config: default
      split: ca_test
      args: default
    metrics:
    - type: rouge
      value: 0.176
      name: Rouge1
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_awesome_billsum_model

This model is a fine-tuned version of [t5-small](https://huggingface.co/t5-small) on the billsum dataset.
It achieves the following results on the evaluation set:
- Loss: 2.4290
- Rouge1: 0.176
- Rouge2: 0.0773
- Rougel: 0.1454
- Rougelsum: 0.1455
- Gen Len: 19.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1 | Rouge2 | Rougel | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:------:|:------:|:------:|:---------:|:-------:|
| No log        | 1.0   | 62   | 2.5195          | 0.1478 | 0.0528 | 0.1197 | 0.1194    | 19.0    |
| No log        | 2.0   | 124  | 2.4660          | 0.1572 | 0.06   | 0.1288 | 0.1287    | 19.0    |
| No log        | 3.0   | 186  | 2.4366          | 0.1691 | 0.0719 | 0.1394 | 0.1396    | 19.0    |
| No log        | 4.0   | 248  | 2.4290          | 0.176  | 0.0773 | 0.1454 | 0.1455    | 19.0    |


### Framework versions

- Transformers 4.23.1
- Pytorch 1.12.1+cu113
- Datasets 2.5.2
- Tokenizers 0.13.1
