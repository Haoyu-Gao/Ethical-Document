---
language: en
tags:
- distilroberta
- sentiment
- NSFW
- inappropriate
- spam
- twitter
- reddit
widget:
- text: I like you. You remind me of me when I was young and stupid.
- text: I see you’ve set aside this special time to humiliate yourself in public.
- text: Have a great weekend! See you next week!
---

# Fine-tuned DistilRoBERTa-base for NSFW Classification

# Model Description 

DistilBERT is a transformer model that performs sentiment analysis. I fine-tuned the model on Reddit posts with the purpose of classifying not safe for work (NSFW) content, specifically text that is considered inappropriate and unprofessional. The model predicts 2 classes, which are NSFW or safe for work (SFW). 

The model is a fine-tuned version of [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert).

It was fine-tuned on 14317 Reddit posts pulled from the (Reddit API) [https://praw.readthedocs.io/en/stable/].

# How to Use 

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classification")
classifier("I see you’ve set aside this special time to humiliate yourself in public.")
```

```python
Output:
[{'label': 'NSFW', 'score': 0.998853325843811}]
```

# Contact

Please reach out to [michelle.li851@duke.edu](mailto:michelle.li851@duke.edu) if you have any questions or feedback.

---