---
language:
- unk
tags:
- autotrain
- text-classification
datasets:
- dafex/autotrain-data-indobert-sentiment-analysis
widget:
- text: I love AutoTrain 🤗
co2_eq_emissions:
  emissions: 1.3428141985163928
---

# Model Trained Using AutoTrain

- Problem type: Binary Classification
- Model ID: 2713480683
- CO2 Emissions (in grams): 1.3428

## Validation Metrics

- Loss: 0.132
- Accuracy: 0.960
- Precision: 0.966
- Recall: 0.973
- AUC: 0.993
- F1: 0.969

## Usage

You can use cURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "I love AutoTrain"}' https://api-inference.huggingface.co/models/dafex/autotrain-indobert-sentiment-analysis-2713480683
```

Or Python API:

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("dafex/autotrain-indobert-sentiment-analysis-2713480683", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("dafex/autotrain-indobert-sentiment-analysis-2713480683", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
```