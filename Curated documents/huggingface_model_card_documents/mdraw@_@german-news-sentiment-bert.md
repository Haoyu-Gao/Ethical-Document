---
{}
---
# German sentiment BERT finetuned on news data

Sentiment analysis model based on https://huggingface.co/oliverguhr/german-sentiment-bert, with additional training on German news texts about migration.

This model is part of the project https://github.com/text-analytics-20/news-sentiment-development, which explores sentiment development in German news articles about migration between 2007 and 2019.

Code for inference (predicting sentiment polarity) on raw text can be found at https://github.com/text-analytics-20/news-sentiment-development/blob/main/sentiment_analysis/bert.py

If you are not interested in polarity but just want to predict discrete class labels (0: positive, 1: negative, 2: neutral), you can also use the model with Oliver Guhr's `germansentiment` package as follows:

First install the package from PyPI:

```bash
pip install germansentiment
```

Then you can use the model in Python:

```python
from germansentiment import SentimentModel

model = SentimentModel('mdraw/german-news-sentiment-bert')

# Examples from our validation dataset
texts = [
    '[...], schwärmt der parteilose Vizebürgermeister und Historiker Christian Matzka von der "tollen Helferszene".',
    'Flüchtlingsheim 11.05 Uhr: Massenschlägerei',
    'Rotterdam habe einen Migrantenanteil von mehr als 50 Prozent.',
]

result = model.predict_sentiment(texts)

print(result)
```

The code above will print:

```python
['positive', 'negative', 'neutral']
```
