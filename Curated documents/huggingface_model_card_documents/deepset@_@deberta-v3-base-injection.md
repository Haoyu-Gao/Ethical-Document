---
language:
- en
- de
license: mit
tags:
- generated_from_trainer
datasets:
- deepset/prompt-injections
metrics:
- accuracy
base_model: microsoft/deberta-v3-base
model-index:
- name: deberta-v3-base-injection
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base-injection

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on the [promp-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0673
- Accuracy: 0.9914

## Model description

This model detects prompt injection attempts and classifies them as "INJECTION". Legitimate requests are classified as "LEGIT". The dataset assumes that legitimate requests are either all sorts of questions of key word searches.

## Intended uses & limitations

If you are using this model to secure your system and it is overly "trigger-happy" to classify requests as injections, consider collecting legitimate examples and retraining the model with the [promp-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset.

## Training and evaluation data

Based in the [promp-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 69   | 0.2353          | 0.9741   |
| No log        | 2.0   | 138  | 0.0894          | 0.9741   |
| No log        | 3.0   | 207  | 0.0673          | 0.9914   |


### Framework versions

- Transformers 4.29.1
- Pytorch 2.0.0+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3