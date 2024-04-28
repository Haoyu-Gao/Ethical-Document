---
language: en
tags:
- distilroberta
- sentiment
- emotion
- twitter
- reddit
widget:
- text: Oh my God, he's lost it. He's totally lost it.
- text: What?
- text: Wow, congratulations! So excited for you!
---

# Fine-tuned DistilRoBERTa-base for Emotion Classification 🤬🤢😀😐😭😲

# Model Description 

DistilRoBERTa-base is a transformer model that performs sentiment analysis. I fine-tuned the model on transcripts from the Friends show with the goal of classifying emotions from text data, specifically dialogue from Netflix shows or movies. The model predicts 6 Ekman emotions and a neutral class. These emotions include anger, disgust, fear, joy, neutrality, sadness, and surprise.

The model is a fine-tuned version of [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/) and [DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base). This model was initially trained on the following table from [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/):

|Name|anger|disgust|fear|joy|neutral|sadness|surprise|
|---|---|---|---|---|---|---|---|
|Crowdflower (2016)|Yes|-|-|Yes|Yes|Yes|Yes|
|Emotion Dataset, Elvis et al. (2018)|Yes|-|Yes|Yes|-|Yes|Yes|
|GoEmotions, Demszky et al. (2020)|Yes|Yes|Yes|Yes|Yes|Yes|Yes|
|ISEAR, Vikash (2018)|Yes|Yes|Yes|Yes|-|Yes|-|
|MELD, Poria et al. (2019)|Yes|Yes|Yes|Yes|Yes|Yes|Yes|
|SemEval-2018, EI-reg, Mohammad et al. (2018) |Yes|-|Yes|Yes|-|Yes|-|

It was fine-tuned on:
|Name|anger|disgust|fear|joy|neutral|sadness|surprise|
|---|---|---|---|---|---|---|---|
|Emotion Lines (Friends)|Yes|Yes|Yes|Yes|Yes|Yes|Yes|

# How to Use 

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
classifier("I love this!")
```

```python
Output:
[{'label': 'joy', 'score': 0.9887555241584778}]
```

# Contact

Please reach out to [michelleli1999@gmail.com](mailto:michelleli1999@gmail.com) if you have any questions or feedback.


# Reference

```
Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
Ashritha R Murthy and K M Anil Kumar 2021 IOP Conf. Ser.: Mater. Sci. Eng. 1110 012009
```