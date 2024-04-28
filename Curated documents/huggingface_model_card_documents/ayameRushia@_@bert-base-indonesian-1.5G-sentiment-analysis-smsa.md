---
language: id
license: mit
tags:
- generated_from_trainer
datasets:
- indonlu
metrics:
- accuracy
widget:
- text: Saya mengapresiasi usaha anda
model-index:
- name: bert-base-indonesian-1.5G-finetuned-sentiment-analysis-smsa
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: indonlu
      type: indonlu
      args: smsa
    metrics:
    - type: accuracy
      value: 0.9373015873015873
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-indonesian-1.5G-finetuned-sentiment-analysis-smsa

This model is a fine-tuned version of [cahya/bert-base-indonesian-1.5G](https://huggingface.co/cahya/bert-base-indonesian-1.5G) on the indonlu dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3390
- Accuracy: 0.9373

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
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.2864        | 1.0   | 688  | 0.2154          | 0.9286   |
| 0.1648        | 2.0   | 1376 | 0.2238          | 0.9357   |
| 0.0759        | 3.0   | 2064 | 0.3351          | 0.9365   |
| 0.044         | 4.0   | 2752 | 0.3390          | 0.9373   |
| 0.0308        | 5.0   | 3440 | 0.4346          | 0.9365   |
| 0.0113        | 6.0   | 4128 | 0.4708          | 0.9365   |
| 0.006         | 7.0   | 4816 | 0.5533          | 0.9325   |
| 0.0047        | 8.0   | 5504 | 0.5888          | 0.9310   |
| 0.0001        | 9.0   | 6192 | 0.5961          | 0.9333   |
| 0.0           | 10.0  | 6880 | 0.5992          | 0.9357   |


### Framework versions

- Transformers 4.14.1
- Pytorch 1.10.0+cu111
- Datasets 1.16.1
- Tokenizers 0.10.3
