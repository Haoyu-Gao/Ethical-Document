---
language:
- ru
tags:
- sentiment
- text-classification
---

# RuBERT for Sentiment Analysis
Short Russian texts sentiment classification

This is a [DeepPavlov/rubert-base-cased-conversational](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational) model trained on aggregated corpus of 351.797 texts.

## Labels
    0: NEUTRAL
    1: POSITIVE
    2: NEGATIVE

## How to use
```python

import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)

@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted
```


## Datasets used for model training

**[RuTweetCorp](https://study.mokoron.com/)**

> Рубцова Ю. Автоматическое построение и анализ корпуса коротких текстов (постов микроблогов) для задачи разработки и тренировки тонового классификатора //Инженерия знаний и технологии семантического веба. – 2012. – Т. 1. – С. 109-116.

**[RuReviews](https://github.com/sismetanin/rureviews)**

> RuReviews: An Automatically Annotated Sentiment Analysis Dataset for Product Reviews in Russian.

**[RuSentiment](http://text-machine.cs.uml.edu/projects/rusentiment/)**

> A. Rogers A. Romanov A. Rumshisky S. Volkova M. Gronas A. Gribov RuSentiment: An Enriched Sentiment Analysis Dataset for Social Media in Russian. Proceedings of COLING 2018.

**[Отзывы о медучреждениях](https://github.com/blanchefort/datasets/tree/master/medical_comments)**

> Датасет содержит пользовательские отзывы о медицинских учреждениях. Датасет собран в мае 2019 года с сайта prodoctorov.ru