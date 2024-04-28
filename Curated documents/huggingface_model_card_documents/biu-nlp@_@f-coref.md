---
language:
- en
license: mit
tags:
- fast
- coreference-resolution
datasets:
- multi_news
- ontonotes
metrics:
- CoNLL
task_categories:
- coreference-resolution
model-index:
- name: biu-nlp/f-coref
  results:
  - task:
      type: coreference-resolution
      name: coreference-resolution
    dataset:
      name: ontonotes
      type: coreference
    metrics:
    - type: CoNLL
      value: 78.5
      name: Avg. F1
---

## F-Coref: Fast, Accurate and Easy to Use Coreference Resolution

[F-Coref](https://arxiv.org/abs/2209.04280) allows to process 2.8K OntoNotes documents in 25 seconds on a V100 GPU (compared to 6 minutes for the [LingMess](https://arxiv.org/abs/2205.12644) model, and to 12 minutes of the popular AllenNLP coreference model) with only a modest drop in accuracy.
The fast speed is achieved through a combination of distillation of a compact model from the LingMess model, and an efficient batching implementation using a technique we call leftover

Please check the [official repository](https://github.com/shon-otmazgin/fastcoref) for more details and updates.

#### Experiments

| Model                 | Runtime | Memory  |
|-----------------------|---------|---------|
| [Joshi et al. (2020)](https://arxiv.org/abs/1907.10529)    | 12:06 | 27.4 |
| [Otmazgin et al. (2022)](https://arxiv.org/abs/2205.12644) | 06:43 | 4.6 |
|      + Batching                                            | 06:00 | 6.6 |
| [Kirstain et al. (2021)](https://arxiv.org/abs/2101.00434) | 04:37 | 4.4 |
| [Dobrovolskii (2021)](https://arxiv.org/abs/2109.04127)    | 03:49 | 3.5 |
| [F-Coref](https://arxiv.org/abs/2209.04280)                | 00:45 | 3.3 |
|      + Batching                                            | 00:35 | 4.5 |
|           + Leftovers batching                             | 00:25 | 4.0 |
The inference time(Min:Sec) and memory(GiB) for each model on 2.8K documents. Average of 3 runs. Hardware, NVIDIA Tesla V100 SXM2.

### Citation

```
@inproceedings{Otmazgin2022FcorefFA,
  title={F-coref: Fast, Accurate and Easy to Use Coreference Resolution},
  author={Shon Otmazgin and Arie Cattan and Yoav Goldberg},
  booktitle={AACL},
  year={2022}
}
```
[F-coref: Fast, Accurate and Easy to Use Coreference Resolution](https://aclanthology.org/2022.aacl-demo.6) (Otmazgin et al., AACL-IJCNLP 2022)