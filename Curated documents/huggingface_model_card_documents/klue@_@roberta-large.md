---
language: ko
tags:
- korean
- klue
mask_token: '[MASK]'
widget:
- text: 대한민국의 수도는 [MASK] 입니다.
---

# KLUE RoBERTa large

Pretrained RoBERTa Model on Korean Language. See [Github](https://github.com/KLUE-benchmark/KLUE) and [Paper](https://arxiv.org/abs/2105.09680) for more details.

## How to use

_NOTE:_ Use `BertTokenizer` instead of RobertaTokenizer. (`AutoTokenizer` will load `BertTokenizer`)

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("klue/roberta-large")
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
```

## BibTeX entry and citation info

```bibtex
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
