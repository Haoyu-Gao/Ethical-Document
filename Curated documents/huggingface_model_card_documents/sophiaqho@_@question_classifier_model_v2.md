---
license: apache-2.0
tags:
- generated_from_trainer
base_model: sophiaqho/question_classifier_model
model-index:
- name: question_classifier_model_v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# question_classifier_model_v2

This model is a fine-tuned version of [sophiaqho/question_classifier_model](https://huggingface.co/sophiaqho/question_classifier_model) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0522

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 159  | 0.0591          |
| No log        | 2.0   | 318  | 0.0509          |
| No log        | 3.0   | 477  | 0.0522          |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.1.0+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0
