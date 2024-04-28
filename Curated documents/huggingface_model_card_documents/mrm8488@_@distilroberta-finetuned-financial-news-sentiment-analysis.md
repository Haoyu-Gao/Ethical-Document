---
license: apache-2.0
tags:
- generated_from_trainer
- financial
- stocks
- sentiment
datasets:
- financial_phrasebank
metrics:
- accuracy
widget:
- text: Operating profit totaled EUR 9.4 mn , down from EUR 11.7 mn in 2004 .
model-index:
- name: distilRoberta-financial-sentiment
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: financial_phrasebank
      type: financial_phrasebank
      args: sentences_allagree
    metrics:
    - type: accuracy
      value: 0.9823008849557522
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilRoberta-financial-sentiment

This model is a fine-tuned version of [distilroberta-base](https://huggingface.co/distilroberta-base) on the financial_phrasebank dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1116
- Accuracy: 0.9823

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
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 255  | 0.1670          | 0.9646   |
| 0.209         | 2.0   | 510  | 0.2290          | 0.9558   |
| 0.209         | 3.0   | 765  | 0.2044          | 0.9558   |
| 0.0326        | 4.0   | 1020 | 0.1116          | 0.9823   |
| 0.0326        | 5.0   | 1275 | 0.1127          | 0.9779   |


### Framework versions

- Transformers 4.10.2
- Pytorch 1.9.0+cu102
- Datasets 1.12.1
- Tokenizers 0.10.3
