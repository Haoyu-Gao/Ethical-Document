---
license: apache-2.0
tags:
- token-classification
datasets:
- conll2003
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: distilroberta-base-ner-conll2003
  results:
  - task:
      type: token-classification
      name: Token Classification
    dataset:
      name: conll2003
      type: conll2003
    metrics:
    - type: precision
      value: 0.9492923423001218
      name: Precision
    - type: recall
      value: 0.9565545901020023
      name: Recall
    - type: f1
      value: 0.9529096297690173
      name: F1
    - type: accuracy
      value: 0.9883096560400111
      name: Accuracy
  - task:
      type: token-classification
      name: Token Classification
    dataset:
      name: conll2003
      type: conll2003
      config: conll2003
      split: validation
    metrics:
    - type: accuracy
      value: 0.9883249976987512
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTEwNzFlMjk0ZDY4NTg2MGQxMDZkM2IyZjdjNDEwYmNiMWY1MWZiNzg1ZjMyZTlkYzQ0MmVmNTZkMjEyMGQ1YiIsInZlcnNpb24iOjF9.zxapWje7kbauQ5-VDNbY487JB5wkN4XqgaLwoX1cSmNfgpp-MPCjqrocxayb1kImbN8CvzOpU1aSfvRfyd5fAw
    - type: precision
      value: 0.9906910190038265
      name: Precision
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWRjMjYyOGQ2MGMwOGE1ODQyNDU1MzZiNWU4MGUzYWVlNjQ3NDhjZDRlZTE0NDlmMGJjZjliZjU2ZmFiZmZiYyIsInZlcnNpb24iOjF9.G_QY9mDkIkllmWPsgmUoVgs-R9XjfYkdJMS8hcyGM-7NXsbigUgZZnhfD0TjDak62UoEplqwSX5r0S4xKPdxBQ
    - type: recall
      value: 0.9916635820847483
      name: Recall
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODE0MDE5ZWMzNTM5MTA1NTI4YzNhNzI2NzVjODIzZWY0OWE2ODJiN2FiNmVkNGVkMTI2ODZiOGEwNTEzNzk2MCIsInZlcnNpb24iOjF9.zenVqRfs8TrKoiIu_QXQJtHyj3dEH97ZDLxUn_UJ2tdW36hpBflgKCJNBvFFkra7bS4cNRfIkwxxCUMWH1ptBg
    - type: f1
      value: 0.9911770619696786
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZWZjY2NiNjZlNDFiODQ3M2JkOWJjNzRlY2FmNjMwNGFkNzFmNTBkOGQ5YTcyZjUzNjAwNDAxMThiNTE5ZThiNiIsInZlcnNpb24iOjF9.c9aD9hycCS-WBaLUb8NKzIpd2LE6xfJrhg3fL9_832RiMq5gcMs9qtarP3Jbo6WbPs_WThr_v4gn7K4Ti-0-CA
    - type: loss
      value: 0.05638007074594498
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNGM3NTQ5ODBhMDcyNjBjMGUxMDgzYjI2NjEwNjM0MjU0MjEzMTRmODA2MjMwZWU1YTQ3OWU2YjUzNTliZTkwMSIsInZlcnNpb24iOjF9.03OwbxrdKm-vg6ia5CBYdEaSCuRbT0pLoEvwpd4NtjydVzo5wzS-pWgY6vH4PlI0ZCTBY0Po0IZSsJulWJttDg
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilroberta-base-ner-conll2003

This model is a fine-tuned version of [distilroberta-base](https://huggingface.co/distilroberta-base) on the conll2003 dataset.

eval F1-Score: **95,29** (CoNLL-03)   
test F1-Score: **90,74** (CoNLL-03)  

eval F1-Score: **95,29** (CoNLL++ / CoNLL-03 corrected)  
test F1-Score: **92,23** (CoNLL++ / CoNLL-03 corrected)  


## Model Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("philschmid/distilroberta-base-ner-conll2003")
model = AutoModelForTokenClassification.from_pretrained("philschmid/distilroberta-base-ner-conll2003")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
example = "My name is Philipp and live in Germany"

nlp(example)

```


## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4.9902376275441704e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 6.0
- mixed_precision_training: Native AMP

### Training results

#### CoNNL2003

It achieves the following results on the evaluation set:
- Loss: 0.0583
- Precision: 0.9493
- Recall: 0.9566
- F1: 0.9529
- Accuracy: 0.9883

It achieves the following results on the test set:
- Loss: 0.2025
- Precision: 0.8999
- Recall: 0.915
- F1: 0.9074
- Accuracy: 0.9741

#### CoNNL++ / CoNLL2003 corrected

It achieves the following results on the evaluation set:
- Loss: 0.0567
- Precision: 0.9493
- Recall: 0.9566
- F1: 0.9529
- Accuracy: 0.9883

It achieves the following results on the test set:
- Loss: 0.1359
- Precision: 0.92
- Recall: 0.9245
- F1: 0.9223
- Accuracy: 0.9785


### Framework versions

- Transformers 4.6.1
- Pytorch 1.8.1+cu101
- Datasets 1.6.2
- Tokenizers 0.10.2
