---
language: zh
tags:
- text-generation
---

## Usage

```
pip install transformers
```

```python
from transformers import CpmAntTokenizer, CpmAntForCausalLM

texts = "今天天气不错，"
model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b")
tokenizer = CpmAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
input_ids = tokenizer(texts, return_tensors="pt")
outputs = model.generate(**input_ids)
output_texts = tokenizer.batch_decode(outputs)

print(output_texts)
```
