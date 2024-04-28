---
language:
- multilingual
- af
- am
- ar
- az
- be
- bg
- bn
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
- fil
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- haw
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
- iw
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
- pa
- pl
- ps
- pt
- ro
- ru
- sd
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
- tr
- uk
- ur
- uz
- vi
- xh
- yi
- yo
- zh
- zu
license: apache-2.0
datasets:
- wikipedia
---

# RemBERT (for classification) 

Pretrained RemBERT model on 110 languages using a masked language modeling (MLM) objective. It was introduced in the paper [Rethinking embedding coupling in pre-trained language models](https://arxiv.org/abs/2010.12821). A direct export of the model checkpoint was first made available in [this repository](https://github.com/google-research/google-research/tree/master/rembert). This version of the checkpoint is lightweight since it is meant to be finetuned for classification and excludes the output embedding weights.

## Model description

RemBERT's main difference with mBERT is that the input and output embeddings are not tied. Instead, RemBERT uses small input embeddings and larger output embeddings. This makes the model more efficient since the output embeddings are discarded during fine-tuning. It is also more accurate, especially when reinvesting the input embeddings' parameters into the core model, as is done on RemBERT. 

## Intended uses & limitations

You should fine-tune this model for your downstream task. It is meant to be a general-purpose model, similar to mBERT. In our [paper](https://arxiv.org/abs/2010.12821), we have successfully applied this model to tasks such as classification, question answering, NER, POS-tagging. For tasks such as text generation you should look at models like GPT2.


## Training data

The RemBERT model was pretrained on multilingual Wikipedia data over 110 languages. The full language list is on [this repository](https://github.com/google-research/google-research/tree/master/rembert)

### BibTeX entry and citation info

```bibtex
@inproceedings{DBLP:conf/iclr/ChungFTJR21,
  author    = {Hyung Won Chung and
               Thibault F{\'{e}}vry and
               Henry Tsai and
               Melvin Johnson and
               Sebastian Ruder},
  title     = {Rethinking Embedding Coupling in Pre-trained Language Models},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=xpFFI\_NtgpW},
  timestamp = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/ChungFTJR21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```