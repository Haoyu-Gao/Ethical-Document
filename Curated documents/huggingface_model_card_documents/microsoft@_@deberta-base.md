---
language: en
license: mit
tags:
- deberta-v1
- fill-mask
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
---

## DeBERTa: Decoding-enhanced BERT with Disentangled Attention

[DeBERTa](https://arxiv.org/abs/2006.03654) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. It outperforms BERT and RoBERTa on  majority of NLU tasks with 80GB training data. 

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more details and updates.


#### Fine-tuning on NLU tasks

We present the dev results on SQuAD 1.1/2.0 and MNLI tasks.

| Model             | SQuAD 1.1 | SQuAD 2.0 | MNLI-m |
|-------------------|-----------|-----------|--------|
| RoBERTa-base      | 91.5/84.6 | 83.7/80.5 | 87.6   |
| XLNet-Large       | -/-       | -/80.2    | 86.8   |
| **DeBERTa-base**  | 93.1/87.2 | 86.2/83.1 | 88.8   |

### Citation

If you find DeBERTa useful for your work, please cite the following paper:

``` latex
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```
