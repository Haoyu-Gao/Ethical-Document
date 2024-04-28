---
language: ko
---

# Electra base model for Korean

* 70GB Korean text dataset and 42000 lower-cased subwords are used
* Check the model performance and other language models for Korean in [github](https://github.com/kiyoungkim1/LM-kor)

```python
from transformers import ElectraTokenizerFast, ElectraModel

tokenizer_electra = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")
model = ElectraModel.from_pretrained("kykim/electra-kor-base")
```