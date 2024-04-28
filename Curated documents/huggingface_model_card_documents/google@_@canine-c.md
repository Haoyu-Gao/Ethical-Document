---
language:
- multilingual
- af
- sq
- ar
- an
- hy
- ast
- az
- ba
- eu
- bar
- be
- bn
- inc
- bs
- br
- bg
- my
- ca
- ceb
- ce
- zh
- cv
- hr
- cs
- da
- nl
- en
- et
- fi
- fr
- gl
- ka
- de
- el
- gu
- ht
- he
- hi
- hu
- is
- io
- id
- ga
- it
- ja
- jv
- kn
- kk
- ky
- ko
- la
- lv
- lt
- roa
- nds
- lm
- mk
- mg
- ms
- ml
- mr
- mn
- min
- ne
- new
- nb
- nn
- oc
- fa
- pms
- pl
- pt
- pa
- ro
- ru
- sco
- sr
- hr
- scn
- sk
- sl
- aze
- es
- su
- sw
- sv
- tl
- tg
- th
- ta
- tt
- te
- tr
- uk
- ud
- uz
- vi
- vo
- war
- cy
- fry
- pnb
- yo
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
---

# CANINE-c (CANINE pre-trained with autoregressive character loss) 

Pretrained CANINE model on 104 languages using a masked language modeling (MLM) objective. It was introduced in the paper [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) and first released in [this repository](https://github.com/google-research/language/tree/master/language/canine). 

What's special about CANINE is that it doesn't require an explicit tokenizer (such as WordPiece or SentencePiece) as other models like BERT and RoBERTa. Instead, it directly operates at a character level: each character is turned into its [Unicode code point](https://en.wikipedia.org/wiki/Code_point#:~:text=For%20Unicode%2C%20the%20particular%20sequence,forming%20a%20self%2Dsynchronizing%20code.). 

This means that input processing is trivial and can typically be accomplished as: 

```
input_ids = [ord(char) for char in text]
```

The ord() function is part of Python, and turns each character into its Unicode code point.

Disclaimer: The team releasing CANINE did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

CANINE is a transformers model pretrained on a large corpus of multilingual data in a self-supervised fashion, similar to BERT. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

* Masked language modeling (MLM): one randomly masks part of the inputs, which the model needs to predict. This model (CANINE-c) is trained with an autoregressive character loss. One masks several character spans within each sequence, which the model then autoregressively predicts.
* Next sentence prediction (NSP): the model concatenates two sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.

This way, the model learns an inner representation of multiple languages that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the CANINE model as inputs.

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=canine) to look for fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation you should look at models like GPT2.

### How to use

Here is how to use this model:

```python
from transformers import CanineTokenizer, CanineModel

model = CanineModel.from_pretrained('google/canine-c')
tokenizer = CanineTokenizer.from_pretrained('google/canine-c')

inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

outputs = model(**encoding) # forward pass
pooled_output = outputs.pooler_output
sequence_output = outputs.last_hidden_state
```

## Training data

The CANINE model was pretrained on on the multilingual Wikipedia data of [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md), which includes 104 languages.


### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2103-06874,
  author    = {Jonathan H. Clark and
               Dan Garrette and
               Iulia Turc and
               John Wieting},
  title     = {{CANINE:} Pre-training an Efficient Tokenization-Free Encoder for
               Language Representation},
  journal   = {CoRR},
  volume    = {abs/2103.06874},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.06874},
  archivePrefix = {arXiv},
  eprint    = {2103.06874},
  timestamp = {Tue, 16 Mar 2021 11:26:59 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-06874.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```