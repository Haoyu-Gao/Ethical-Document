---
language:
- en
license: apache-2.0
tags:
- text-classification
- emotion
- pytorch
datasets:
- emotion
metrics:
- Accuracy, F1 Score
thumbnail: https://avatars3.githubusercontent.com/u/32437151?s=460&u=4ec59abc8d21d5feea3dab323d23a5860e6996a4&v=4
model-index:
- name: bhadresh-savani/distilbert-base-uncased-emotion
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: emotion
      type: emotion
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 0.927
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYzQxOGRmMjFlZThmZWViNjNmNGMzMTdjMGNjYjg1YWUzOTI0ZDlmYjRhYWMzMDA3Yjg2N2FiMTdmMzk0ZjJkOSIsInZlcnNpb24iOjF9.mOqr-hgNrnle7WCPy3Mo7M3fITFppn5gjpNagGMf_TZfB6VZnPKfZ51UkNFQlBtUlcm0U8vwPkF79snxwvCoDw
    - type: precision
      value: 0.8880230732280744
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYjZiN2NjNTkyN2M3ZWM2ZDZiNDk1OWZhN2FmNTAwZDIzMmQ3NTU2Yjk2MTgyNjJmMTNjYTYzOTc1NDdhYTljYSIsInZlcnNpb24iOjF9.0rWHmCZ2PyZ5zYkSeb_tFdQG9CHS5PdpOZ9kOfrIzEXyZ968daayaOJi2d6iO84fnauE5hZiIAUPsx24Vr4nBA
    - type: precision
      value: 0.927
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZmRhNWM1NDQ4ZjkyYjAxYjQ5MzQzMDA1ZDIzYWU3YTE4NTI2ZTMwYWI2ZWQ4NzQ3YzJkODYzMmZhZDI1NGRlNCIsInZlcnNpb24iOjF9.NlII1s42Mr_DMzPEoR0ntyh5cDW0405TxVkWhCgXLJTFAdnivH54-zZY4av1U5jHPTeXeWwZrrrbMwHCRBkoCw
    - type: precision
      value: 0.9272902840835793
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODhkNmM5NmYyMzA4MjkwOTllZDgyMDQ1NzZkN2QzOTAyOTMyNGFlZTU4NzM5NmM5NWQ1YmUxYmRmNjA5YjhhNCIsInZlcnNpb24iOjF9.oIn1KT-BOpFNLXiKL29frMvgHhWZMHWc9Q5WgeR7UaMEO7smkK8J3j5HAMy17Ktjv2dh783-f76N6gyJ_NewCg
    - type: recall
      value: 0.8790126653780703
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYjhlNzczNDY2NDVlM2UwMjAzOWQxYTAyNWZkNGZlYmNjODNiZTEzMTcxNTE3MTAxNjNkOTFiMmRiMzViMzJmZiIsInZlcnNpb24iOjF9.AXp7omMuUZFJ6mzAVTQPMke7QoUtoi4RJSSE7Xbnp2pNi7y-JtznKdm---l6RfqcHPlI0jWr7TVGoFsWZ64YAg
    - type: recall
      value: 0.927
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjEyYmZiZDQ4MzM1ZmQ2ZmJhZWU4OTVkNmViYjA5NzhiN2MxODE0MzUxZTliZTk0MzViZDAyNGU4MDFjYjM1MSIsInZlcnNpb24iOjF9.9lazxLXbPOdwhqoYtIudwRwjfNVZnUu7KvGRklRP_RAoQStAzgmWMIrT3ckX_d5_6bKZH9fIdujUn5Qz-baKBw
    - type: recall
      value: 0.927
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWVhMzY0YTA4YmQzYTg4YTBiMzQ5YzRiZWJhMjM1NjUzZGQxZmQ5M2NkZDcyNTQ0ZmJjN2NkY2ZiYjg0OWI0ZCIsInZlcnNpb24iOjF9.QgTv726WCTyvrEct0NM8Zpc3vUnDbIwCor9EH941-zpJtuWr-xpdZzYZFJfILkVA0UUn1y6Jz_ABfkfBeyZTBg
    - type: f1
      value: 0.8825061528287809
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzQzZTJkMDAwOTUwMzY3ZjI2MjIxYjlmZTg3YTdhNTc4ZjYyMmQ2NDQzM2FmYzk3OGEzNjhhMTk3NTQ3OTlhNyIsInZlcnNpb24iOjF9.hSln1KfKm0plK7Qao9vlubFtAl1M7_UYHNM6La9gEZlW_apnU1Mybz03GT2XZORgOVPe9JmgygvZByxQhpsYBw
    - type: f1
      value: 0.927
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzljODQ3NjE3MDRkODE3ZjFlZmY5MjYyOGJlNDQ4YzdlZGRiMTI5OGZiZWM2ODkyZjMyZWQ3MTkzYWU5YThkOCIsInZlcnNpb24iOjF9.7qfBw39fv22jSIJoY71DkOVr9eBB-srhqSi09bCcUC7Huok4O2Z_vB7gO_Rahh9sFgKVu1ZATusjTmOLQr0fBw
    - type: f1
      value: 0.926876082854655
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjJhN2UzODgxOWQ0Y2E3YTcwZTQxMDE0ZWRmYThjOWVhYWQ1YjBhMzk0YWUxNzE2ZjFhNWM5ZmE2ZmI1YTczYSIsInZlcnNpb24iOjF9.nZW0dBdLmh_FgNw6GaITvSJFX-2C_Iku3NanU8Rip7FSiRHozKPAjothdQh9MWQnq158ZZGPPVIjtyIvuTSqCw
    - type: loss
      value: 0.17403268814086914
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMTVjZmFiOGQwZGY1OTU5YWFkNGZjMTlhOGI4NjE3MGI4ZDhkODcxYmJiYTQ3NWNmMWM0ODUyZDI1MThkYTY3ZSIsInZlcnNpb24iOjF9.OYz5BI3Lz8LgjAqVnD6NcrG3UAG0D3wjKJ7G5298RRGaNpb621ycisG_7UYiWixY7e2RJafkfRiplmkdczIFDQ
---
# Distilbert-base-uncased-emotion

## Model description:
[Distilbert](https://arxiv.org/abs/1910.01108) is created with knowledge distillation during the pre-training phase which reduces the size of a BERT model by 40%, while retaining 97% of its language understanding. It's smaller, faster than Bert and any other Bert-based model.

[Distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) finetuned on the emotion dataset using HuggingFace Trainer with below Hyperparameters
```
 learning rate 2e-5, 
 batch size 64,
 num_train_epochs=8,
```

## Model Performance Comparision on Emotion Dataset from Twitter:

| Model | Accuracy | F1 Score |  Test Sample per Second |
| --- | --- | --- | --- |
| [Distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) | 93.8 | 93.79 | 398.69 |
| [Bert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion) | 94.05 | 94.06 | 190.152 |
| [Roberta-base-emotion](https://huggingface.co/bhadresh-savani/roberta-base-emotion) | 93.95 | 93.97| 195.639 |
| [Albert-base-v2-emotion](https://huggingface.co/bhadresh-savani/albert-base-v2-emotion) | 93.6 | 93.65 | 182.794 |

## How to Use the model:
```python
from transformers import pipeline
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
prediction = classifier("I love using transformers. The best part is wide range of support and its easy to use", )
print(prediction)

"""
Output:
[[
{'label': 'sadness', 'score': 0.0006792712374590337}, 
{'label': 'joy', 'score': 0.9959300756454468}, 
{'label': 'love', 'score': 0.0009452480007894337}, 
{'label': 'anger', 'score': 0.0018055217806249857}, 
{'label': 'fear', 'score': 0.00041110432357527316}, 
{'label': 'surprise', 'score': 0.0002288572577526793}
]]
"""
```

## Dataset:
[Twitter-Sentiment-Analysis](https://huggingface.co/nlp/viewer/?dataset=emotion).

## Training procedure
[Colab Notebook](https://github.com/bhadreshpsavani/ExploringSentimentalAnalysis/blob/main/SentimentalAnalysisWithDistilbert.ipynb)

## Eval results
```json
{
'test_accuracy': 0.938,
 'test_f1': 0.937932884041714,
 'test_loss': 0.1472451239824295,
 'test_mem_cpu_alloc_delta': 0,
 'test_mem_cpu_peaked_delta': 0,
 'test_mem_gpu_alloc_delta': 0,
 'test_mem_gpu_peaked_delta': 163454464,
 'test_runtime': 5.0164,
 'test_samples_per_second': 398.69
 }
```

## Reference:
* [Natural Language Processing with Transformer By Lewis Tunstall, Leandro von Werra, Thomas Wolf](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/)