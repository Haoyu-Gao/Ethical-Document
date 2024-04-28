---
language: mn
tags:
- bert
- mongolian
- cased
---

# BERT-LARGE-MONGOLIAN-CASED
[Link to Official Mongolian-BERT repo](https://github.com/tugstugi/mongolian-bert)

## Model description
This repository contains pre-trained Mongolian [BERT](https://arxiv.org/abs/1810.04805) models trained by [tugstugi](https://github.com/tugstugi), [enod](https://github.com/enod) and [sharavsambuu](https://github.com/sharavsambuu).
Special thanks to [nabar](https://github.com/nabar) who provided 5x TPUs.

This repository is based on the following open source projects: [google-research/bert](https://github.com/google-research/bert/),
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese).

#### How to use

```python
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-large-mongolian-cased', use_fast=False)
model = AutoModelForMaskedLM.from_pretrained('tugstugi/bert-large-mongolian-cased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = 'Монгол улсын [MASK] Улаанбаатар хотоос ярьж байна.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])

## output ##
# {'sequence': 'Монгол улсын нийслэл Улаанбаатар хотоос ярьж байна.', 'score': 0.9779232740402222, 'token': 1176, 'token_str': 'нийслэл'}
# {'sequence': 'Монгол улсын Нийслэл Улаанбаатар хотоос ярьж байна.', 'score': 0.015034765936434269, 'token': 4059, 'token_str': 'Нийслэл'}
# {'sequence': 'Монгол улсын Ерөнхийлөгч Улаанбаатар хотоос ярьж байна.', 'score': 0.0021413620561361313, 'token': 325, 'token_str': 'Ерөнхийлөгч'}
# {'sequence': 'Монгол улсын ерөнхийлөгч Улаанбаатар хотоос ярьж байна.', 'score': 0.0008035294013097882, 'token': 1215, 'token_str': 'ерөнхийлөгч'}
# {'sequence': 'Монгол улсын нийслэлийн Улаанбаатар хотоос ярьж байна.', 'score': 0.0006434018723666668, 'token': 356, 'token_str': 'нийслэлийн'}
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
