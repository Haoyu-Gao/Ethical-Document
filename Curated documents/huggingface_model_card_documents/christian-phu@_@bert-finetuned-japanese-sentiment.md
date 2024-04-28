---
language:
- ja
license: cc-by-sa-4.0
tags:
- generated_from_trainer
metrics:
- accuracy
pipeline_tag: text-classification
model-index:
- name: bert-finetuned-japanese-sentiment
  results: []
---
  
# bert-finetuned-japanese-sentiment

This model is a fine-tuned version of [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) on product amazon reviews japanese dataset.

## Model description

Model Train for amazon reviews Japanese sentence sentiments.

Sentiment analysis is a common task in natural language processing. It consists of classifying the polarity of a given text at the sentence or document level. For instance, the sentence "The food is good" has a positive sentiment, while the sentence "The food is bad" has a negative sentiment.

In this model, we fine-tuned a BERT model on a Japanese sentiment analysis dataset. The dataset contains 20,000 sentences extracted from Amazon reviews. Each sentence is labeled as positive, neutral, or negative. The model was trained for 5 epochs with a batch size of 16.

## Training and evaluation data

- Epochs: 6
- Training Loss: 0.087600
- Validation Loss: 1.028876
- Accuracy: 0.813202
- Precision: 0.712440
- Recall: 0.756031
- F1: 0.728455

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 0
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 6

### Framework versions

- Transformers 4.27.4
- Pytorch 2.0.0+cu118
- Tokenizers 0.13.2