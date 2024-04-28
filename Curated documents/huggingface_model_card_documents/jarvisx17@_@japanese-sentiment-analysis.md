---
language: ja
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
widget:
- text: 🤗セグメント利益は、前期比8.3％増の24億28百万円となった
model-index:
- name: Japanese-sentiment-analysis
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# japanese-sentiment-analysis

This model was trained from scratch on the chABSA dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0001
- Accuracy: 1.0
- F1: 1.0

## Model description

Model Train for Japanese sentence sentiments.

## Intended uses & limitations

The model was trained on chABSA Japanese dataset.
DATASET link : https://www.kaggle.com/datasets/takahirokubo0/chabsa

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10


## Usage

You can use cURL to access this model:

Python API:

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("jarvisx17/japanese-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")

inputs = tokenizer("I love AutoNLP", return_tensors="pt")

outputs = model(**inputs)
```

### Training results



### Framework versions

- Transformers 4.24.0
- Pytorch 1.12.1+cu113
- Datasets 2.7.0
- Tokenizers 0.13.2

### Dependencies
- !pip install fugashi
- !pip install unidic_lite
