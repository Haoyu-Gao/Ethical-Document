---
language: mn
tags:
- bert
- mongolian
- uncased
---

# BERT-LARGE-MONGOLIAN-UNCASED
[Link to Official Mongolian-BERT repo](https://github.com/tugstugi/mongolian-bert)

## Model description
This repository contains pre-trained Mongolian [BERT](https://arxiv.org/abs/1810.04805) models trained by [tugstugi](https://github.com/tugstugi), [enod](https://github.com/enod) and [sharavsambuu](https://github.com/sharavsambuu).
Special thanks to [nabar](https://github.com/nabar) who provided 5x TPUs.

This repository is based on the following open source projects: [google-research/bert](https://github.com/google-research/bert/),
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese).

#### How to use

```python
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-large-mongolian-uncased', use_fast=False)
model = AutoModelForMaskedLM.from_pretrained('tugstugi/bert-large-mongolian-uncased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = 'Монгол улсын [MASK] Улаанбаатар хотоос ярьж байна.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])

## output ##
# {'sequence': 'монгол улсын нийслэл улаанбаатар хотоос ярьж байна.', 'score': 0.7867621183395386, 'token': 849, 'token_str': 'нийслэл'}
# {'sequence': 'монгол улсын ерөнхийлөгч улаанбаатар хотоос ярьж байна.', 'score': 0.14303277432918549, 'token': 244, 'token_str': 'ерөнхийлөгч'}
# {'sequence': 'монгол улсын ерөнхийлөгчийг улаанбаатар хотоос ярьж байна.', 'score': 0.011642335914075375, 'token': 8373, 'token_str': 'ерөнхийлөгчийг'}
# {'sequence': 'монгол улсын иргэд улаанбаатар хотоос ярьж байна.', 'score': 0.006592822726815939, 'token': 247, 'token_str': 'иргэд'}
# {'sequence': 'монгол улсын нийслэлийг улаанбаатар хотоос ярьж байна.', 'score': 0.006165097933262587, 'token': 15501, 'token_str': 'нийслэлийг'}
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
