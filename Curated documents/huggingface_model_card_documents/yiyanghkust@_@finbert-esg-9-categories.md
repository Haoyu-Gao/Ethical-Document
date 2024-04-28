---
language: en
tags:
- financial-text-analysis
- esg
- environmental-social-corporate-governance
widget:
- text: 'For 2002, our total net emissions were approximately 60 million metric tons
    of CO2 equivalents for all businesses and operations we have ﬁnancial interests
    in, based on its equity share in those businesses and operations. '
---

ESG analysis can help investors determine a business' long-term sustainability and identify associated risks. **FinBERT-esg-9-categories** is a FinBERT model fine-tuned on about 14,000 manually annotated sentences from firms' ESG reports and annual reports. 

**finbert-esg-9-categories** classifies a text into nine fine-grained ESG topics: *Climate Change, Natural Capital, Pollution & Waste, Human Capital, Product Liability, Community Relations, Corporate Governance, Business Ethics & Values, and Non-ESG*. This model complements [**finbert-esg**](https://huggingface.co/yiyanghkust/finbert-esg) which classifies a text into four coarse-grained ESG themes (*E, S, G or None*). 

Detailed description of the nine fine-grained ESG topic definition, some examples for each topic, training sample, and the model’s performance can be found [**here**](https://www.allenhuang.org/uploads/2/6/5/5/26555246/esg_9-class_descriptions.pdf).

**Input**: A text.

**Output**: Climate Change, Natural Capital, Pollution & Waste, Human Capital, Product Liability, Community Relations, Corporate Governance, Business Ethics & Values, or Non-ESG.

# How to use 
You can use this model with Transformers pipeline for fine-grained ESG 9 categories classification.

```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories',num_labels=9)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

results = nlp('For 2002, our total net emissions were approximately 60 million metric tons of CO2 equivalents for all businesses 
               and operations we have ﬁnancial interests in, based on its equity share in those businesses and operations.')
print(results) # [{'label': 'Climate Change', 'score': 0.9955655932426453}]
```


If you use the model in your academic work, please cite the following paper: 

Huang, Allen H., Hui Wang, and Yi Yang. "FinBERT: A Large Language Model for Extracting Information from Financial Text." *Contemporary Accounting Research* (2022).