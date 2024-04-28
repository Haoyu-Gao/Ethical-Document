---
language:
- vi
license: mit
tags:
- sentiment
- classification
widget:
- text: Không thể nào đẹp hơn
- text: Quá phí tiền, mà không đẹp
- text: Cái này giá ổn không nhỉ?
---

[**GitHub Homepage**](https://github.com/wonrax/phobert-base-vietnamese-sentiment)

A model fine-tuned for sentiment analysis based on [vinai/phobert-base](https://huggingface.co/vinai/phobert-base).

Labels:
- NEG: Negative
- POS: Positive
- NEU: Neutral

Dataset: [30K e-commerce reviews](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst)

## Usage
```python
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

# Just like PhoBERT: INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Đây là mô_hình rất hay , phù_hợp với điều_kiện và như cầu của nhiều người .'  

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    out = model(input_ids)
    print(out.logits.softmax(dim=-1).tolist())
    # Output:
    # [[0.002, 0.988, 0.01]]
    #     ^      ^      ^
    #    NEG    POS    NEU
```
