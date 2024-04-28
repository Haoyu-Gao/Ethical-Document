---
language:
- en
tags:
- generated_from_trainer
- crypto
- sentiment
- analysis
pipeline_tag: text-classification
base_model: ProsusAI/finbert
model-index:
- name: CryptoBERT
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# CryptoBERT

This model is a fine-tuned version of [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on the Custom Crypto Market Sentiment dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3823

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

tokenizer = BertTokenizer.from_pretrained("kk08/CryptoBERT")
model = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "Bitcoin (BTC) touches $29k, Ethereum (ETH) Set To Explode, RenQ Finance (RENQ) Crosses Massive Milestone"
result = classifier(text)
print(result)

```
```
[{'label': 'LABEL_1', 'score': 0.9678454399108887}]
```
## Model description

This model fine-tunes the [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), which is a pre-trained NLP model to analyze the sentiment of the financial text. 
CryptoBERT model fine-tunes this by training the model as a downstream task on Custom Crypto Sentiment data to predict whether the given text related to the Crypto market is
Positive (LABEL_1) or Negative (LABEL_0).

## Intended uses & limitations

The model can perform well on Crypto-related data. The main limitation is that the fine-tuning was done using only a small corpus of data

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.4077        | 1.0   | 27   | 0.4257          |
| 0.2048        | 2.0   | 54   | 0.2479          |
| 0.0725        | 3.0   | 81   | 0.3068          |
| 0.0028        | 4.0   | 108  | 0.4120          |
| 0.0014        | 5.0   | 135  | 0.3566          |
| 0.0007        | 6.0   | 162  | 0.3495          |
| 0.0006        | 7.0   | 189  | 0.3645          |
| 0.0005        | 8.0   | 216  | 0.3754          |
| 0.0004        | 9.0   | 243  | 0.3804          |
| 0.0004        | 10.0  | 270  | 0.3823          |


### Framework versions

- Transformers 4.28.0
- Pytorch 2.0.0+cu118
- Datasets 2.11.0
- Tokenizers 0.13.3