---
language:
- af
- am
- ar
- as
- az
- be
- bg
- bn
- bo
- bs
- ca
- ceb
- co
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- haw
- he
- hi
- hmn
- hr
- ht
- hu
- hy
- id
- ig
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lb
- lo
- lt
- lv
- mg
- mi
- mk
- ml
- mn
- mr
- ms
- mt
- my
- ne
- nl
- false
- ny
- or
- pa
- pl
- pt
- ro
- ru
- rw
- si
- sk
- sl
- sm
- sn
- so
- sq
- sr
- st
- su
- sv
- sw
- ta
- te
- tg
- th
- tk
- tl
- tr
- tt
- ug
- uk
- ur
- uz
- vi
- wo
- xh
- yi
- yo
- zh
- zu
license: apache-2.0
tags:
- bert
- sentence_embedding
- multilingual
- google
- sentence-similarity
datasets:
- CommonCrawl
- Wikipedia
pipeline_tag: sentence-similarity
---

# LaBSE

## Model description

Language-agnostic BERT Sentence Encoder (LaBSE) is a BERT-based model trained for sentence embedding for 109 languages. The pre-training process combines masked language modeling with translation language modeling. The model is useful for getting multilingual sentence embeddings and for bi-text retrieval.

- Model: [HuggingFace's model hub](https://huggingface.co/setu4993/LaBSE).
- Paper: [arXiv](https://arxiv.org/abs/2007.01852).
- Original model: [TensorFlow Hub](https://tfhub.dev/google/LaBSE/2).
- Blog post: [Google AI Blog](https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html).
- Conversion from TensorFlow to PyTorch: [GitHub](https://github.com/setu4993/convert-labse-tf-pt).

This is migrated from the v2 model on the TF Hub, which uses dict-based input. The embeddings produced by both the versions of the model are [equivalent](https://github.com/setu4993/convert-labse-tf-pt/blob/ec3a019159a54ed6493181a64486c2808c01f216/tests/test_conversion.py#L31).

## Usage

Using the model:

```python
import torch
from transformers import BertModel, BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()

english_sentences = [
    "dog",
    "Puppies are nice.",
    "I enjoy taking long walks along the beach with my dog.",
]
english_inputs = tokenizer(english_sentences, return_tensors="pt", padding=True)

with torch.no_grad():
    english_outputs = model(**english_inputs)
```

To get the sentence embeddings, use the pooler output:

```python
english_embeddings = english_outputs.pooler_output
```

Output for other languages:

```python
italian_sentences = [
    "cane",
    "I cuccioli sono carini.",
    "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.",
]
japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]
italian_inputs = tokenizer(italian_sentences, return_tensors="pt", padding=True)
japanese_inputs = tokenizer(japanese_sentences, return_tensors="pt", padding=True)

with torch.no_grad():
    italian_outputs = model(**italian_inputs)
    japanese_outputs = model(**japanese_inputs)

italian_embeddings = italian_outputs.pooler_output
japanese_embeddings = japanese_outputs.pooler_output
```

For similarity between sentences, an L2-norm is recommended before calculating the similarity:

```python
import torch.nn.functional as F


def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


print(similarity(english_embeddings, italian_embeddings))
print(similarity(english_embeddings, japanese_embeddings))
print(similarity(italian_embeddings, japanese_embeddings))
```

## Details

Details about data, training, evaluation and performance metrics are available in the [original paper](https://arxiv.org/abs/2007.01852).

### BibTeX entry and citation info

```bibtex
@misc{feng2020languageagnostic,
      title={Language-agnostic BERT Sentence Embedding},
      author={Fangxiaoyu Feng and Yinfei Yang and Daniel Cer and Naveen Arivazhagan and Wei Wang},
      year={2020},
      eprint={2007.01852},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
