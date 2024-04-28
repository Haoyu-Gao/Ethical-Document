---
license: apache-2.0
datasets:
- amazon_polarity
base_model: distilbert-base-uncased
model-index:
- name: distilbert-base-uncased-finetuned-sentiment-amazon
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: amazon_polarity
      type: sentiment
      args: default
    metrics:
    - type: accuracy
      value: 0.961
      name: Accuracy
    - type: loss
      value: 0.116
      name: Loss
    - type: f1
      value: 0.96
      name: F1
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: amazon_polarity
      type: amazon_polarity
      config: amazon_polarity
      split: test
    metrics:
    - type: accuracy
      value: 0.94112
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzlmMzdhYjNmN2U0NDBkM2U5ZDgwNzc3YjE1OGE4MWUxMDY1N2U0ODc0YzllODE5ODIyMzdkOWFhNzVjYmI5MyIsInZlcnNpb24iOjF9.3nlcLa4IpPQtklp7_U9XzC__Q_JVf_cWs6JVVII8trhX5zg_q9HEyQOQs4sRf6O-lIJg8zb3mgobZDJShuSJAQ
    - type: precision
      value: 0.9321570625232675
      name: Precision
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZjI2MDY4NGNlYjhjMGMxODBiNTc2ZjM5YzY1NjkxNTU4MDA2ZDIyY2QyZjUyZmE4YWY0N2Y1ODU5YTc2ZDM0NiIsInZlcnNpb24iOjF9.egEikTa2UyHV6SAGkHJKaa8FRwGHoZmJRCmqUQaJqeF5yxkz2V-WeCHoWDrCXsHCbXEs8UhLlyo7Lr83BPfkBg
    - type: recall
      value: 0.95149
      name: Recall
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiM2E3M2Y3MDU4ZTM2YjdlZjQ0NTY3NGYwMmQ3NTk5ZmZkZWUwZWZiZDZjNjk2ZWE5MmY4MmZiM2FmN2U2M2QyNCIsInZlcnNpb24iOjF9.4VNbiWRmSee4cxuIZ5m7bN30i4BpK7xtHQ1BF8AuFIXkWQgzOmGdX35bLhLGWW8KL3ClA4RDPVBKYCIrw0YUBw
    - type: auc
      value: 0.9849019044624999
      name: AUC
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYTkwODk2ZTUwOTViNjBhYTU0ODk1MDA3MDY1NDkyZDc2YmRlNTQzNDE3YmE3YTVkYjNhN2JmMDAxZWQ0NjUxZSIsInZlcnNpb24iOjF9.YEr6OhqOL7QnqYqjUTQFMdkgU_uS1-vVnkJtn_-1UwSoX754UV_bL9S9KSH3DX4m5QFoRXdZxfeOocm1JbzaCA
    - type: f1
      value: 0.9417243188138998
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzIyMmViNTQ3ZGU0M2I5ZmRjOGI1OWMwZGEwYmE5OGU5YTZlZTkzZjdkOTQ4YzJmOTc2MDliMDY4NDQ1NGRlNyIsInZlcnNpb24iOjF9.p05MGHTfHTAzp4u-qfiIn6Zmh5c3TW_uwjXWgbb982pL_oCILQb6jFXqhPpWXL321fPye7qaUVbGhcTJd8sdCA
    - type: loss
      value: 0.16342754662036896
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzgxMDc4M2IxYjhkNjRhZmYyNzY1MTNkNzhmYjk2NmU1NjFiOTk1NDIzNzI1ZGU3MDYyYjQ2YmQ1NTI2N2NhMyIsInZlcnNpb24iOjF9.Zuf0nzn8XdvwRChKtE9CwJ0pgpc6Zey6oTR3jRiSkvNY2sNbo2bvAgFimGzgGYkDvRvYkTCXzCyxdb27l3QnAg
---

# distilbert-sentiment

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on a subset of the [amazon-polarity dataset](https://huggingface.co/datasets/amazon_polarity).

<b>[Update 10/10/23]</b> The model has been retrained on a larger part of the dataset with an improvement on the loss, f1 score and accuracy. It achieves the following results on the evaluation set:
- Loss: 0.116
- Accuracy: 0.961
- F1_score: 0.960

## Model description

This sentiment classifier has been trained on 360_000 samples for the training set, 40_000 samples for the validation set and 40_000 samples for the test set.

## Intended uses & limitations
```python
from transformers import pipeline

# Create the pipeline
sentiment_classifier = pipeline('text-classification', model='AdamCodd/distilbert-base-uncased-finetuned-sentiment-amazon')

# Now you can use the pipeline to get the sentiment
result = sentiment_classifier("This product doesn't fit me at all.")
print(result)
#[{'label': 'negative', 'score': 0.9994848966598511}]
```

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 1270
- optimizer: AdamW with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 150
- num_epochs: 2
- weight_decay: 0.01

### Training results
(Previous results before retraining from the model evaluator)
| key | value |
| --- | ----- |
| eval_accuracy | 0.94112 |
| eval_auc | 0.9849 |
| eval_f1_score | 0.9417 |
| eval_precision | 0.9321 |
| eval_recall | 0.95149 |
### Framework versions

- Transformers 4.34.0
- Pytorch lightning 2.0.9
- Tokenizers 0.14.0

If you want to support me, you can [here](https://ko-fi.com/adamcodd).