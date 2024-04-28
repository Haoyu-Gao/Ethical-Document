---
language: en
tags:
- neural-search-query-classification
- neural-search
widget:
- text: what did you eat in lunch?
---
# KEYWORD STATEMENT VS QUESTION CLASSIFIER FOR NEURAL SEARCH


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")

model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")
```
Trained to add the feature for classifying queries between Question Query vs Statement Query using classification in [Haystack](https://github.com/deepset-ai/haystack/issues/611)





