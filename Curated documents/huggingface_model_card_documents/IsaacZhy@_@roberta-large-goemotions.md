---
license: mit
tags:
- generated_from_trainer
datasets:
- go_emotions
metrics:
- f1
- accuracy
model-index:
- name: pretrained_model
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: go_emotions
      type: go_emotions
      config: simplified
      split: validation
      args: simplified
    metrics:
    - type: f1
      value: 0.586801681970308
      name: F1
    - type: accuracy
      value: 0.4821231109472908
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pretrained_model

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the go_emotions dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0568
- F1: 0.5868
- Roc Auc: 0.7616
- Accuracy: 0.4821

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
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1     | Roc Auc | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:------:|:-------:|:--------:|
| 0.1205        | 1.0   | 679  | 0.0865          | 0.5632 | 0.7347  | 0.4458   |
| 0.0859        | 2.0   | 1358 | 0.0829          | 0.5717 | 0.7378  | 0.4521   |
| 0.0727        | 3.0   | 2037 | 0.0827          | 0.5897 | 0.7523  | 0.4753   |
| 0.0629        | 4.0   | 2716 | 0.0857          | 0.5808 | 0.7535  | 0.4652   |
| 0.0568        | 5.0   | 3395 | 0.0904          | 0.5868 | 0.7616  | 0.4821   |
| 0.0423        | 6.0   | 4074 | 0.0989          | 0.5806 | 0.7682  | 0.4724   |
| 0.0344        | 7.0   | 4753 | 0.1079          | 0.5736 | 0.7657  | 0.4650   |
| 0.0296        | 8.0   | 5432 | 0.1158          | 0.5637 | 0.7649  | 0.4504   |
| 0.0206        | 9.0   | 6111 | 0.1200          | 0.5674 | 0.7689  | 0.4486   |
| 0.0177        | 10.0  | 6790 | 0.1240          | 0.5728 | 0.7737  | 0.4547   |


### Framework versions

- Transformers 4.26.0
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2
