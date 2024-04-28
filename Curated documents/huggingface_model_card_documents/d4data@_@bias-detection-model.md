---
language:
- en
tags:
- Text Classification
co2_eq_emissions: 0.319355
widget:
- text: Nevertheless, Trump and other Republicans have tarred the protests as havens
    for terrorists intent on destroying property.
  example_title: Biased example 1
- text: Billie Eilish issues apology for mouthing an anti-Asian derogatory term in
    a resurfaced video.
  example_title: Biased example 2
- text: Christians should make clear that the perpetuation of objectionable vaccines
    and the lack of alternatives is a kind of coercion.
  example_title: Biased example 3
- text: There have been a protest by a group of people
  example_title: Non-Biased example 1
- text: While emphasizing he’s not singling out either party, Cohen warned about the
    danger of normalizing white supremacist ideology.
  example_title: Non-Biased example 2
---

## About the Model
An English sequence classification model, trained on MBAD Dataset to detect bias and fairness in sentences (news articles). This model was built on top of distilbert-base-uncased model and trained for 30 epochs with a batch size of 16, a learning rate of 5e-5, and a maximum sequence length of 512.

- Dataset : MBAD Data
- Carbon emission 0.319355 Kg

| Train Accuracy | Validation Accuracy | Train loss | Test loss |
|---------------:| -------------------:| ----------:|----------:|
|          76.97 |               62.00 |       0.45 |      0.96 |

## Usage
The easiest way is to load the inference api from huggingface and second method is through the pipeline object offered by transformers library.
```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer) # cuda = 0,1 based on gpu availability
classifier("The irony, of course, is that the exhibit that invites people to throw trash at vacuuming Ivanka Trump lookalike reflects every stereotype feminists claim to stand against, oversexualizing Ivanka’s body and ignoring her hard work.")
```

## Author
This model is part of the Research topic "Bias and Fairness in AI" conducted by Deepak John Reji, Shaina Raza. If you use this work (code, model or dataset), please star at:
> Bias & Fairness in AI, (2022), GitHub repository, <https://github.com/dreji18/Fairness-in-AI>

