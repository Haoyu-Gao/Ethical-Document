---
license: apache-2.0
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
base_model: google/vit-base-patch16-224-in21k
model-index:
- name: rorshark-vit-base
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: imagefolder
      type: imagefolder
      config: default
      split: train
      args: default
    metrics:
    - type: accuracy
      value: 0.9922928709055877
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# rorshark-vit-base

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0393
- Accuracy: 0.9923

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.0597        | 1.0   | 368  | 0.0546          | 0.9865   |
| 0.2009        | 2.0   | 736  | 0.0531          | 0.9865   |
| 0.0114        | 3.0   | 1104 | 0.0418          | 0.9904   |
| 0.0998        | 4.0   | 1472 | 0.0425          | 0.9904   |
| 0.1244        | 5.0   | 1840 | 0.0393          | 0.9923   |


### Framework versions

- Transformers 4.36.0.dev0
- Pytorch 2.1.1+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0
