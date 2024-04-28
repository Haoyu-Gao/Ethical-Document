---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- imdb
metrics:
- accuracy
model-index:
- name: distilbert-imdb
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: imdb
      type: imdb
      args: plain_text
    metrics:
    - type: accuracy
      value: 0.928
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-imdb

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the imdb dataset (training notebook is [here](https://huggingface.co/lvwerra/distilbert-imdb/blob/main/distilbert-imdb-training.ipynb)).
It achieves the following results on the evaluation set:
- Loss: 0.1903
- Accuracy: 0.928

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
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.2195        | 1.0   | 1563 | 0.1903          | 0.928    |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.10.0+cu111
- Datasets 1.17.0
- Tokenizers 0.10.3
