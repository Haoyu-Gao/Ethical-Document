---
language: mn
tags:
- bert
- mongolian
- cased
---

# BERT-BASE-MONGOLIAN-CASED
[Link to Official Mongolian-BERT repo](https://github.com/tugstugi/mongolian-bert)

## Model description
This repository contains pre-trained Mongolian [BERT](https://arxiv.org/abs/1810.04805) models trained by [tugstugi](https://github.com/tugstugi), [enod](https://github.com/enod) and [sharavsambuu](https://github.com/sharavsambuu).
Special thanks to [nabar](https://github.com/nabar) who provided 5x TPUs.

This repository is based on the following open source projects: [google-research/bert](https://github.com/google-research/bert/),
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese).

#### How to use

```python
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-base-mongolian-cased', use_fast=False)
model = AutoModelForMaskedLM.from_pretrained('tugstugi/bert-base-mongolian-cased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = '[MASK] хот Монгол улсын нийслэл.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])

## output ##
# {'sequence': 'Улаанбаатар хот Монгол улсын нийслэл.', 'score': 0.826970100402832, 'token': 281, 'token_str': 'Улаанбаатар'}
# {'sequence': 'Нийслэл хот Монгол улсын нийслэл.', 'score': 0.06551621109247208, 'token': 4059, 'token_str': 'Нийслэл'}
# {'sequence': 'Эрдэнэт хот Монгол улсын нийслэл.', 'score': 0.0264141745865345, 'token': 2229, 'token_str': 'Эрдэнэт'}
# {'sequence': 'Дархан хот Монгол улсын нийслэл.', 'score': 0.017083868384361267, 'token': 1646, 'token_str': 'Дархан'}
# {'sequence': 'УБ хот Монгол улсын нийслэл.', 'score': 0.010854342952370644, 'token': 7389, 'token_str': 'УБ'}
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
