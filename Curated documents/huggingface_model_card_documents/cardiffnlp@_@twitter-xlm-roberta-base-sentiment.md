---
language: multilingual
widget:
- text: 🤗
- text: T'estimo! ❤️
- text: I love you!
- text: I hate you 🤮
- text: Mahal kita!
- text: 사랑해!
- text: 난 너가 싫어
- text: 😍😍😍
---


# twitter-XLM-roBERTa-base for Sentiment Analysis

This is a multilingual XLM-roBERTa-base model trained on ~198M tweets and finetuned for sentiment analysis. The sentiment fine-tuning was done on 8 languages (Ar, En, Fr, De, Hi, It, Sp, Pt) but it can be used for more languages (see paper for details).

- Paper: [XLM-T: A Multilingual Language Model Toolkit for Twitter](https://arxiv.org/abs/2104.12250). 
- Git Repo: [XLM-T official repository](https://github.com/cardiffnlp/xlm-t).

This model has been integrated into the [TweetNLP library](https://github.com/cardiffnlp/tweetnlp).

## Example Pipeline
```python
from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("T'estimo!")
```
```
[{'label': 'Positive', 'score': 0.6600581407546997}]
```

## Full classification example

```python
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Good night 😊"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Good night 😊"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)

# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

```

Output: 

```
1) Positive 0.7673
2) Neutral 0.2015
3) Negative 0.0313
```

### Reference
```
@inproceedings{barbieri-etal-2022-xlm,
    title = "{XLM}-{T}: Multilingual Language Models in {T}witter for Sentiment Analysis and Beyond",
    author = "Barbieri, Francesco  and
      Espinosa Anke, Luis  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.27",
    pages = "258--266"
}

```

