---
language:
- ru
- en
license: mit
tags:
- russian
- fill-mask
- pretraining
- embeddings
- masked-lm
- tiny
- feature-extraction
- sentence-similarity
widget:
- text: Миниатюрная модель для [MASK] разных задач.
pipeline_tag: fill-mask
---
This is a very small distilled version of the [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) model for Russian and English (45 MB, 12M parameters). There is also an **updated version of this model**, [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2), with a larger vocabulary and better quality on practically all Russian NLU tasks.

This model is useful if you want to fine-tune it for a relatively simple Russian task (e.g. NER or sentiment classification), and you care more about speed and size than about accuracy. It is approximately x10 smaller and faster than a base-sized BERT. Its `[CLS]` embeddings can be used as a sentence representation aligned between Russian and English. 

It was trained on the [Yandex Translate corpus](https://translate.yandex.ru/corpus), [OPUS-100](https://huggingface.co/datasets/opus100) and [Tatoeba](https://huggingface.co/datasets/tatoeba), using MLM loss (distilled from [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)), translation ranking loss, and `[CLS]` embeddings distilled from [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), [rubert-base-cased-sentence](https://huggingface.co/DeepPavlov/rubert-base-cased-sentence), Laser and USE.

There is a more detailed [description in Russian](https://habr.com/ru/post/562064/). 

Sentence embeddings can be produced as follows:

```python
# pip install transformers sentencepiece
import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
# model.cuda()  # uncomment it if you have a GPU

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

print(embed_bert_cls('привет мир', model, tokenizer).shape)
# (312,)
```