---
language:
- multilingual
- af
- am
- ar
- as
- az
- be
- bg
- bn
- br
- bs
- ca
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
- he
- hi
- hr
- hu
- hy
- id
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
- lo
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- my
- ne
- nl
- false
- om
- or
- pa
- pl
- ps
- pt
- ro
- ru
- sa
- sd
- si
- sk
- sl
- so
- sq
- sr
- su
- sv
- sw
- ta
- te
- th
- tl
- tr
- ug
- uk
- ur
- uz
- vi
- xh
- yi
- zh
license: mit
---

# XLM-RoBERTa-XL (xlarge-sized model) 

XLM-RoBERTa-XL model pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages. It was introduced in the paper [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/abs/2105.00572) by Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau and first released in [this repository](https://github.com/pytorch/fairseq/tree/master/examples/xlmr). 

Disclaimer: The team releasing XLM-RoBERTa-XL did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

XLM-RoBERTa-XL is a extra large multilingual version of RoBERTa. It is pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages. 

RoBERTa is a transformers model pretrained on a large corpus in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts.

More precisely, it was pretrained with the Masked language modeling (MLM) objective. Taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.

This way, the model learns an inner representation of 100 languages that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the XLM-RoBERTa-XL model as inputs.

## Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?search=xlm-roberta-xl) to look for fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation, you should look at models like GPT2.

## Usage

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='facebook/xlm-roberta-xl')
>>> unmasker("Europe is a <mask> continent.")

[{'score': 0.08562745153903961,
  'token': 38043,
  'token_str': 'living',
  'sequence': 'Europe is a living continent.'},
 {'score': 0.0799778401851654,
  'token': 103494,
  'token_str': 'dead',
  'sequence': 'Europe is a dead continent.'},
 {'score': 0.046154674142599106,
  'token': 72856,
  'token_str': 'lost',
  'sequence': 'Europe is a lost continent.'},
 {'score': 0.04358183592557907,
  'token': 19336,
  'token_str': 'small',
  'sequence': 'Europe is a small continent.'},
 {'score': 0.040570393204689026,
  'token': 34923,
  'token_str': 'beautiful',
  'sequence': 'Europe is a beautiful continent.'}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('facebook/xlm-roberta-xl')
model = AutoModelForMaskedLM.from_pretrained("facebook/xlm-roberta-xl")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)
```

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2105-00572,
  author    = {Naman Goyal and
               Jingfei Du and
               Myle Ott and
               Giri Anantharaman and
               Alexis Conneau},
  title     = {Larger-Scale Transformers for Multilingual Masked Language Modeling},
  journal   = {CoRR},
  volume    = {abs/2105.00572},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.00572},
  eprinttype = {arXiv},
  eprint    = {2105.00572},
  timestamp = {Wed, 12 May 2021 15:54:31 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-00572.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```