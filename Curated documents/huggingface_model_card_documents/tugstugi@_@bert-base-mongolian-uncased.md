---
language: mn
tags:
- bert
- mongolian
- uncased
---

# BERT-BASE-MONGOLIAN-UNCASED
[Link to Official Mongolian-BERT repo](https://github.com/tugstugi/mongolian-bert)

## Model description
This repository contains pre-trained Mongolian [BERT](https://arxiv.org/abs/1810.04805) models trained by [tugstugi](https://github.com/tugstugi), [enod](https://github.com/enod) and [sharavsambuu](https://github.com/sharavsambuu).
Special thanks to [nabar](https://github.com/nabar) who provided 5x TPUs.

This repository is based on the following open source projects: [google-research/bert](https://github.com/google-research/bert/),
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese).

#### How to use

```python
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-base-mongolian-uncased', use_fast=False)
model = AutoModelForMaskedLM.from_pretrained('tugstugi/bert-base-mongolian-uncased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = 'Миний [MASK] хоол идэх нь тун чухал.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])

## output ##
#{'sequence': 'миний хувьд хоол идэх нь тун чухал.', 'score': 0.7889143824577332, 'token': 126, 'token_str': 'хувьд'}
#{'sequence': 'миний бодлоор хоол идэх нь тун чухал.', 'score': 0.18616807460784912, 'token': 6106, 'token_str': 'бодлоор'}
#{'sequence': 'миний зүгээс хоол идэх нь тун чухал.', 'score': 0.004825591575354338, 'token': 761, 'token_str': 'зүгээс'}
#{'sequence': 'миний биед хоол идэх нь тун чухал.', 'score': 0.0015743684489279985, 'token': 3010, 'token_str': 'биед'}
#{'sequence': 'миний тухайд хоол идэх нь тун чухал.', 'score': 0.0014919431414455175, 'token': 1712, 'token_str': 'тухайд'}
```


## Training data
Mongolian Wikipedia and the 700 million word Mongolian news data set  [[Pretraining Procedure](https://github.com/tugstugi/mongolian-bert#pre-training)]

### BibTeX entry and citation info

```bibtex
@misc{mongolian-bert,
  author = {Tuguldur, Erdene-Ochir and Gunchinish, Sharavsambuu and Bataa, Enkhbold},
  title = {BERT Pretrained Models on Mongolian Datasets},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tugstugi/mongolian-bert/}}
}
```
