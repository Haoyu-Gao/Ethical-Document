---
language:
- en
tags:
- text
- nlp
- correction
pipeline_tag: text-classification
---


# Model Trained Using AutoNLP

- Problem type: Multi-class Classification
- Model ID: 492513457
- CO2 Emissions (in grams): 5.527544460835904

## Validation Metrics

- Loss: 0.07609463483095169
- Accuracy: 0.9735624586913417
- Macro F1: 0.9736173135739408
- Micro F1: 0.9735624586913417
- Weighted F1: 0.9736173135739408
- Macro Precision: 0.9737771415197378
- Micro Precision: 0.9735624586913417
- Weighted Precision: 0.9737771415197378
- Macro Recall: 0.9735624586913417
- Micro Recall: 0.9735624586913417
- Weighted Recall: 0.9735624586913417


## Usage

You can use CURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "Is this text really worth it?"}' https://api-inference.huggingface.co/models/wajidlinux99/gibberish-text-detector
```

Or Python API:

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("wajidlinux99/gibberish-text-detector", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("wajidlinux99/gibberish-text-detector", use_auth_token=True)

inputs = tokenizer("Is this text really worth it?", return_tensors="pt")

outputs = model(**inputs)
```

# Original Repository

***madhurjindal/autonlp-Gibberish-Detector-492513457