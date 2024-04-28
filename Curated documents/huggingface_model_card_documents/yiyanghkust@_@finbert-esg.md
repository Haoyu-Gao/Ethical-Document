---
language: en
tags:
- financial-text-analysis
- esg
- environmental-social-corporate-governance
widget:
- text: 'Rhonda has been volunteering for several years for a variety of charitable
    community programs. '
---

ESG analysis can help investors determine a business' long-term sustainability and identify associated risks. FinBERT-ESG is a FinBERT model fine-tuned on 2,000 manually annotated sentences from firms' ESG reports and annual reports.  

**Input**: A financial text.

**Output**: Environmental, Social, Governance or None.

# How to use 
You can use this model with Transformers pipeline for ESG classification.
```python
# tested in transformers==4.18.0 
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp('Rhonda has been volunteering for several years for a variety of charitable community programs.')
print(results) # [{'label': 'Social', 'score': 0.9906041026115417}]

```

Visit [FinBERT.AI](https://finbert.ai/) for more details on the recent development of FinBERT.

If you use the model in your academic work, please cite the following paper:

Huang, Allen H., Hui Wang, and Yi Yang. "FinBERT: A Large Language Model for Extracting Information from Financial Text." *Contemporary Accounting Research* (2022).

