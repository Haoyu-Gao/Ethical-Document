---
license: apache-2.0
tags:
- generated_from_trainer
- image-classification
- pytorch
datasets:
- food101
metrics:
- accuracy
model-index:
- name: food101_outputs
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: food-101
      type: food101
      args: default
    metrics:
    - type: accuracy
      value: 0.8912871287128713
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# nateraw/food

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the nateraw/food101 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4501
- Accuracy: 0.8913

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 128
- eval_batch_size: 128
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.8271        | 1.0   | 592  | 0.6070          | 0.8562   |
| 0.4376        | 2.0   | 1184 | 0.4947          | 0.8691   |
| 0.2089        | 3.0   | 1776 | 0.4876          | 0.8747   |
| 0.0882        | 4.0   | 2368 | 0.4639          | 0.8857   |
| 0.0452        | 5.0   | 2960 | 0.4501          | 0.8913   |


### Framework versions

- Transformers 4.9.0.dev0
- Pytorch 1.9.0+cu102
- Datasets 1.9.1.dev0
- Tokenizers 0.10.3
