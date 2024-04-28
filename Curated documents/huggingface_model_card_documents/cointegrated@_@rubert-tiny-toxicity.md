---
language:
- ru
tags:
- russian
- classification
- toxicity
- multilabel
widget:
- text: Иди ты нафиг!
---
This is the [cointegrated/rubert-tiny](https://huggingface.co/cointegrated/rubert-tiny) model fine-tuned for classification of toxicity and inappropriateness for short informal Russian texts, such as comments in social networks. 

The problem is formulated as multilabel classification with the following classes:
- `non-toxic`: the text does NOT contain insults, obscenities, and threats, in the sense of the [OK ML Cup](https://cups.mail.ru/ru/tasks/1048) competition.
- `insult`
- `obscenity`
- `threat`
- `dangerous`: the text is inappropriate, in the sense of [Babakov et.al.](https://arxiv.org/abs/2103.05345), i.e. it can harm the reputation of the speaker.

A text can be considered safe if it is BOTH `non-toxic` and NOT `dangerous`. 

## Usage

The function below estimates the probability that the text is either toxic OR dangerous:
```python
# !pip install transformers sentencepiece --quiet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()
    
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

print(text2toxicity('я люблю нигеров', True))
# 0.9350118728093193

print(text2toxicity('я люблю нигеров', False))
# [0.9715758  0.0180863  0.0045551  0.00189755 0.9331106 ]

print(text2toxicity(['я люблю нигеров', 'я люблю африканцев'], True))
# [0.93501186 0.04156357]

print(text2toxicity(['я люблю нигеров', 'я люблю африканцев'], False))
# [[9.7157580e-01 1.8086294e-02 4.5550885e-03 1.8975559e-03 9.3311059e-01]
#  [9.9979788e-01 1.9048342e-04 1.5297388e-04 1.7452303e-04 4.1369814e-02]]
```

## Training

The model has been trained on the joint dataset of [OK ML Cup](https://cups.mail.ru/ru/tasks/1048) and [Babakov et.al.](https://arxiv.org/abs/2103.05345) with `Adam` optimizer, the learning rate of `1e-5`, and batch size of `64` for `15` epochs. A text was considered inappropriate if its inappropriateness score was higher than 0.8, and appropriate - if it was lower than 0.2. The per-label ROC AUC on the dev set is:
```
non-toxic  : 0.9937
insult     : 0.9912
obscenity  : 0.9881
threat     : 0.9910
dangerous  : 0.8295
```