---
language:
- en
- fr
- ro
- de
- multilingual
license: apache-2.0
tags:
- summarization
- translation
datasets:
- c4
---

## [t5-small](https://huggingface.co/t5-small) exported to the ONNX format

## Model description

[T5](https://huggingface.co/docs/transformers/model_doc/t5#t5) is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.

For more information, please take a look at the original paper.

Paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)

Authors: *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu*


## Usage example

You can use this model with Transformers *pipeline*.

```python
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("optimum/t5-small")
model = ORTModelForSeq2SeqLM.from_pretrained("optimum/t5-small")
translator = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
results = translator("My name is Eustache and I have a pet raccoon")
print(results)
```
