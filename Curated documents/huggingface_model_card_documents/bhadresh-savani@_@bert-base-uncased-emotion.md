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
- name: bhadresh-savani/bert-base-uncased-emotion
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
      value: 0.9265
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWQzNzA2MTFkY2RkNDMxYTFhOGUzMTdiZTgwODA3ODdmZTVhNTVjOTAwMGM5NjU1OGY0MjMzZWU0OTU2MzY1YiIsInZlcnNpb24iOjF9.f6iWK0iyU8_g32W2oMfh1ChevMsl0StI402cB6DNzJCYj9xywTnFltBY36jAJFDRK41HXdMnPMl64Bynr-Q9CA
    - type: precision
      value: 0.8859601677706858
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNTc2ZjRmMzYzNTE0ZDQ1ZDdkYWViYWNhZDhkOTE2ZDhmMDFjZmZiZjRkZWVlMzQ3MWE4NDNlYzlmM2I4ZGM2OCIsInZlcnNpb24iOjF9.jR-gFrrBIAfiYV352RDhK3nzgqIgNCPd55OhIcCfVdVAWHQSZSJXhFyg8yChC7DwoVmUQy1Ya-d8Hflp7Wi-AQ
    - type: precision
      value: 0.9265
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMDAyMWZjZTM5NWNjNTcyMWQzMWQyNDcyN2RlZTQyZTM4ZDQ4Y2FlNzM2OTZkMzM3YzI4YTAwNzg4MGNjZmZjZCIsInZlcnNpb24iOjF9.cmkuDmhhETKIKAL81K28oiO889sZ0hvEpZ6Ep7dW_KB9VOTFs15BzFY9vwcpdXQDugWBbB2g7r3FUgRLwIEpAg
    - type: precision
      value: 0.9265082039990273
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMTA2NzY2NTJmZTExZWM3OGIzYzg3ZDM3Y2I5MTU3Mjg3Y2NmZGEyMjFmNjExZWM3ZDFjNzdhOTZkNTYwYWQxYyIsInZlcnNpb24iOjF9.DJgeA6ZovHoxgCqhzilIzafet8uN3-Xbx1ZYcEEc4jXzFbRtErE__QHGaaSaUQEzPp4BAztp1ageOaBoEmXSDg
    - type: recall
      value: 0.879224648382427
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZGU3MmQ1Yjg5OGJlYTE1NWJmNGVjY2ExMDZiZjVjYmVkOGYxYWFkOTVlMDVjOWVhZGFjOGFkYzcwMGIyMTAyZCIsInZlcnNpb24iOjF9.jwgaNEBSQENlx3vojBi1WKJOQ7pSuP4Iyw4kKPsq9IUaW-Ah8KdgPV9Nm2DY1cwEtMayvVeIVmQ3Wo8PORDRAg
    - type: recall
      value: 0.9265
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNDE3OWQ0ZGZjNzAxY2I0NGMxNDU0OWE1OGM2N2Q3OTUwYWI0NmZjMDQ3MDc0NDA4YTc2NDViM2Y0ZTMyMjYyZCIsInZlcnNpb24iOjF9.Ihc61PSO3K63t5hUSAve4Gt1tC8R_ZruZo492dTD9CsKOF10LkvrCskJJaOATjFJgqb3FFiJ8-nDL9Pa3HF-Dg
    - type: recall
      value: 0.9265
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzJkYTg5YjA0YTBlNDY3ZjFjZWIzOWVhYjI4Y2YxM2FhMmUwMDZlZTE0NTIzNjMxMjE3NzgwNGFjYTkzOWM1YyIsInZlcnNpb24iOjF9.LlBX4xTjKuTX0NPK0jYzYDXRVnUEoUKVwIHfw5xUzaFgtF4wuqaYV7F0VKoOd3JZxzxNgf7JzeLof0qTquE9Cw
    - type: f1
      value: 0.8821398657055098
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNTE4OThiMmE0NDEzZjBkY2RmZWNjMGI3YWNmNTFjNTY5NjIwNjFkZjk1ZjIxMjI4M2ZiZGJhYzJmNzVhZTU1NSIsInZlcnNpb24iOjF9.gzYyUbO4ycvP1RXnrKKZH3E8ym0DjwwUFf4Vk9j0wrg2sWIchjmuloZz0SLryGqwHiAV8iKcSBWWy61Q480XAw
    - type: f1
      value: 0.9265
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZGM2Y2E0NjMyNmJhMTE4NjYyMjI2MTJlZjUzNmRmY2U3Yjk3ZGUyYzU2OWYzMWM2ZjY4ZTg0OTliOTY3YmI2MSIsInZlcnNpb24iOjF9.hEz_yExs6LV0RBpFBoUbnAQZHitxN57HodCJpDx0yyW6dQwWaza0JxdO-kBf8JVBK8JyISkNgOYskBY5LD4ZDQ
    - type: f1
      value: 0.9262425173620311
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZmMyY2NhNTRhOGMwM2M5OTQxNDQ0NjRkZDdiMDExMWFkMmI4MmYwZGQ1OGRiYmRjMmE2YTc0MGZmMWMwN2Q4MSIsInZlcnNpb24iOjF9.ljbb2L4R08NCGjcfuX1878HRilJ_p9qcDJpWhsu-5EqWCco80e9krb7VvIJV0zBfmi7Z3C2qGGRsfsAIhtQ5Dw
    - type: loss
      value: 0.17315374314785004
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZmQwN2I2Nzg4OWU1ODE5NTBhMTZiMjljMjJhN2JiYmY0MTkzMTA1NmVhMGU0Y2Y0NjgyOTU3ZjgyYTc3ODE5NCIsInZlcnNpb24iOjF9.EEp3Gxm58ab-9335UGQEk-3dFQcMRgJgViI7fpz7mfY2r5Pg-AOel5w4SMzmBM-hiUFwStgxe5he_kG2yPGFCw
---
# bert-base-uncased-emotion

## Model description:

[Bert](https://arxiv.org/abs/1810.04805) is a Transformer Bidirectional Encoder based Architecture trained on MLM(Mask Language Modeling) objective

[bert-base-uncased](https://huggingface.co/bert-base-uncased) finetuned on the emotion dataset using HuggingFace Trainer with below training parameters
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
classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
prediction = classifier("I love using transformers. The best part is wide range of support and its easy to use", )
print(prediction)

"""
output:
[[
{'label': 'sadness', 'score': 0.0005138228880241513}, 
{'label': 'joy', 'score': 0.9972520470619202}, 
{'label': 'love', 'score': 0.0007443308713845909}, 
{'label': 'anger', 'score': 0.0007404946954920888}, 
{'label': 'fear', 'score': 0.00032938539516180754}, 
{'label': 'surprise', 'score': 0.0004197491507511586}
]]
"""
```

## Dataset:
[Twitter-Sentiment-Analysis](https://huggingface.co/nlp/viewer/?dataset=emotion).

## Training procedure
[Colab Notebook](https://github.com/bhadreshpsavani/ExploringSentimentalAnalysis/blob/main/SentimentalAnalysisWithDistilbert.ipynb)
follow the above notebook by changing the model name from distilbert to bert

## Eval results
```json
{
 'test_accuracy': 0.9405,
 'test_f1': 0.9405920712282673,
 'test_loss': 0.15769127011299133,
 'test_runtime': 10.5179,
 'test_samples_per_second': 190.152,
 'test_steps_per_second': 3.042
 }
```

## Reference:
* [Natural Language Processing with Transformer By Lewis Tunstall, Leandro von Werra, Thomas Wolf](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/)