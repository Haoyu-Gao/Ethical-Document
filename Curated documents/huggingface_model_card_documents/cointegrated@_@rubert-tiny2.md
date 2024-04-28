---
language:
- ru
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
- sentence-transformers
- transformers
pipeline_tag: sentence-similarity
widget:
- text: Миниатюрная модель для [MASK] разных задач.
---
This is an updated version of [cointegrated/rubert-tiny](https://huggingface.co/cointegrated/rubert-tiny): a small Russian BERT-based encoder with high-quality sentence embeddings. This [post in Russian](https://habr.com/ru/post/669674/) gives more details.

The differences from the previous version include:
- a larger vocabulary: 83828 tokens instead of 29564;
- larger supported sequences: 2048 instead of 512;
- sentence embeddings approximate LaBSE closer than before;
- meaningful segment embeddings (tuned on the NLI task)
- the model is focused only on Russian. 

The model should be used as is to produce sentence embeddings (e.g. for KNN classification of short texts) or fine-tuned for a downstream task.

Sentence embeddings can be produced as follows:

```python
# pip install transformers sentencepiece
import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
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

Alternatively, you can use the model with `sentence_transformers`:
```Python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('cointegrated/rubert-tiny2')
sentences = ["привет мир", "hello world", "здравствуй вселенная"]
embeddings = model.encode(sentences)
print(embeddings)
```