---
language:
- en
license: openrail
library_name: transformers
datasets:
- imdb
metrics:
- accuracy
pipeline_tag: text-classification
---

## Model description
This model is a fine-tuned version of the [bert-base-uncased](https://huggingface.co/transformers/model_doc/bert.html) model to classify the sentiment of movie reviews into one of two
categories: negative(label 0), positive(label 1).

## How to use

You can use the model with the following code.

```python
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
model_path = "JiaqiLee/imdb-finetuned-bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("The movie depicted well the psychological battles that Harry Vardon fought within himself, from his childhood trauma of being evicted to his own inability to break that glass ceiling that prevents him from being accepted as an equal in English golf society."))
```

## Training data
The training data comes from HuggingFace [IMDB dataset](https://huggingface.co/datasets/imdb). We use 90% of the `train.csv` data to train the model and the remaining 10% for evaluation.

## Evaluation results

The model achieves 0.91 classification accuracy in IMDB test dataset.