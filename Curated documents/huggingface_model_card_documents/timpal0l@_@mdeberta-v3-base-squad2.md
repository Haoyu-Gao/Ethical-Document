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
- 'no'
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
tags:
- deberta
- deberta-v3
- mdeberta
- question-answering
- qa
- multilingual
datasets:
- squad_v2
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
---
## This model can be used for Extractive QA
It has been finetuned for 3 epochs on [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/).

## Usage
```python
from transformers import pipeline

qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
question = "Where do I live?"
context = "My name is Tim and I live in Sweden."
qa_model(question = question, context = context)
# {'score': 0.975547730922699, 'start': 28, 'end': 36, 'answer': ' Sweden.'}
```

## Evaluation on SQuAD2.0 dev set
```bash
{
    "epoch": 3.0,
    "eval_HasAns_exact": 79.65587044534414,
    "eval_HasAns_f1": 85.91387795001529,
    "eval_HasAns_total": 5928,
    "eval_NoAns_exact": 82.10260723296888,
    "eval_NoAns_f1": 82.10260723296888,
    "eval_NoAns_total": 5945,
    "eval_best_exact": 80.8809904826076,
    "eval_best_exact_thresh": 0.0,
    "eval_best_f1": 84.00551406448994,
    "eval_best_f1_thresh": 0.0,
    "eval_exact": 80.8809904826076,
    "eval_f1": 84.00551406449004,
    "eval_samples": 12508,
    "eval_total": 11873,
    "train_loss": 0.7729689576483615,
    "train_runtime": 9118.953,
    "train_samples": 134891,
    "train_samples_per_second": 44.377,
    "train_steps_per_second": 0.925
}
``` 
## DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing

[DeBERTa](https://arxiv.org/abs/2006.03654) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With those two improvements, DeBERTa out perform RoBERTa on a majority of NLU tasks with 80GB training data. 

In [DeBERTa V3](https://arxiv.org/abs/2111.09543), we further improved the efficiency of DeBERTa using ELECTRA-Style pre-training with Gradient Disentangled Embedding Sharing. Compared to DeBERTa,  our V3 version significantly improves the model performance on downstream tasks.  You can find more technique details about the new model from our [paper](https://arxiv.org/abs/2111.09543).

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more implementation details and updates.

mDeBERTa is multilingual version of DeBERTa which use the same structure as DeBERTa and was trained with CC100 multilingual data.
The mDeBERTa V3 base model comes with 12 layers and a hidden size of 768. It has 86M backbone parameters  with a vocabulary containing 250K tokens which introduces 190M parameters in the Embedding layer.  This model was trained using the 2.5T CC100 data as XLM-R.