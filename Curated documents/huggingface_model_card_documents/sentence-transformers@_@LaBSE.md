---
language:
- multilingual
- af
- sq
- am
- ar
- hy
- as
- az
- eu
- be
- bn
- bs
- bg
- my
- ca
- ceb
- zh
- co
- hr
- cs
- da
- nl
- en
- eo
- et
- fi
- fr
- fy
- gl
- ka
- de
- el
- gu
- ht
- ha
- haw
- he
- hi
- hmn
- hu
- is
- ig
- id
- ga
- it
- ja
- jv
- kn
- kk
- km
- rw
- ko
- ku
- ky
- lo
- la
- lv
- lt
- lb
- mk
- mg
- ms
- ml
- mt
- mi
- mr
- mn
- ne
- false
- ny
- or
- fa
- pl
- pt
- pa
- ro
- ru
- sm
- gd
- sr
- st
- sn
- si
- sk
- sl
- so
- es
- su
- sw
- sv
- tl
- tg
- ta
- tt
- te
- th
- bo
- tr
- tk
- ug
- uk
- ur
- uz
- vi
- cy
- wo
- xh
- yi
- yo
- zu
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
---

# LaBSE
This is a port of the [LaBSE](https://tfhub.dev/google/LaBSE/1) model to PyTorch. It can be used to map 109 languages to a shared vector space.


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings = model.encode(sentences)
print(embeddings)
```



## Evaluation Results



For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/LaBSE)



## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
  (3): Normalize()
)
```

## Citing & Authors

Have a look at [LaBSE](https://tfhub.dev/google/LaBSE/1) for the respective publication that describes LaBSE.

