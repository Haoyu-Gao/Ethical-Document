---
language:
- en
tags:
- toxic comments classification
licenses:
- cc-by-nc-sa
---

## Toxicity Classification Model

This model is trained for toxicity classification task. The dataset used for training is the merge of the English parts of the three datasets by **Jigsaw** ([Jigsaw 2018](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), [Jigsaw 2019](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), [Jigsaw 2020](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)), containing around 2 million examples. We split it into two parts and fine-tune a RoBERTa model ([RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)) on it. The classifiers perform closely on the test set of the first Jigsaw competition, reaching the **AUC-ROC** of 0.98 and **F1-score** of 0.76.

## How to use
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# load tokenizer and model weights
tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

# prepare the input
batch = tokenizer.encode('you are amazing', return_tensors='pt')

# inference
model(batch)
```

## Licensing Information

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png