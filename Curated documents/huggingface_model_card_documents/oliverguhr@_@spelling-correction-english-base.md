---
language:
- en
license: mit
metrics:
- cer
widget:
- text: lets do a comparsion
  example_title: '1'
- text: Their going to be here so0n
  example_title: '2'
- text: ze shop is cloed due to covid 19
  example_title: '3'
---

This is an experimental model that should fix your typos and punctuation.
If you like to run your own experiments or train for a different language, have a look at [the code](https://github.com/oliverguhr/spelling).


## Model description

This is a proof of concept spelling correction model for English.

## Intended uses & limitations

This project is work in progress, be aware that the model can produce artefacts. 
You can test the model using the pipeline-interface:

```python
from transformers import pipeline

fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")

print(fix_spelling("lets do a comparsion",max_length=2048))
```
