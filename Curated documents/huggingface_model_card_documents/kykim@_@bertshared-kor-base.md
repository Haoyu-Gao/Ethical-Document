---
language: ko
---

# Bert base model for Korean

* 70GB Korean text dataset and 42000 lower-cased subwords are used
* Check the model performance and other language models for Korean in [github](https://github.com/kiyoungkim1/LM-kor)

```python
# only for pytorch in transformers
from transformers import BertTokenizerFast, EncoderDecoderModel

tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")
model = EncoderDecoderModel.from_pretrained("kykim/bertshared-kor-base")
```