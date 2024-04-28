---
language: ko
---

# Bert base model for Korean

## Update

- Update at 2021.11.17 : Add Native Support for BERT Tokenizer (works with AutoTokenizer, pipeline)

---

* 70GB Korean text dataset and 42000 lower-cased subwords are used
* Check the model performance and other language models for Korean in [github](https://github.com/kiyoungkim1/LM-kor)

```python
from transformers import pipeline

pipe = pipeline('text-generation', model='beomi/kykim-gpt3-kor-small_based_on_gpt2')
print(pipe("안녕하세요! 오늘은"))
# [{'generated_text': '안녕하세요! 오늘은 제가 요즘 사용하고 있는 클렌징워터를 소개해드리려고 해요! 바로 이 제품!! 바로 이'}]
```
