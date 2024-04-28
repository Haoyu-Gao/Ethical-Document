---
language: en
license: apache-2.0
tags:
- luke
- named entity recognition
- entity typing
- relation classification
- question answering
thumbnail: https://github.com/studio-ousia/luke/raw/master/resources/luke_logo.png
---

## LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention

**LUKE** (**L**anguage **U**nderstanding with **K**nowledge-based
**E**mbeddings) is a new pre-trained contextualized representation of words and
entities based on transformer. LUKE treats words and entities in a given text as
independent tokens, and outputs contextualized representations of them. LUKE
adopts an entity-aware self-attention mechanism that is an extension of the
self-attention mechanism of the transformer, and considers the types of tokens
(words or entities) when computing attention scores.

LUKE achieves state-of-the-art results on five popular NLP benchmarks including
**[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)** (extractive
question answering),
**[CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)** (named entity
recognition), **[ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)**
(cloze-style question answering),
**[TACRED](https://nlp.stanford.edu/projects/tacred/)** (relation
classification), and
**[Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)**
(entity typing).

Please check the [official repository](https://github.com/studio-ousia/luke) for
more details and updates.

This is the LUKE base model with 12 hidden layers, 768 hidden size. The total number
of parameters in this model is 253M. It is trained using December 2018 version of
Wikipedia.

### Experimental results

The experimental results are provided as follows:

| Task                           | Dataset                                                                      | Metric | LUKE-large        | luke-base | Previous SOTA                                                             |
| ------------------------------ | ---------------------------------------------------------------------------- | ------ | ----------------- | --------- | ------------------------------------------------------------------------- |
| Extractive Question Answering  | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)                    | EM/F1  | **90.2**/**95.4** | 86.1/92.3 | 89.9/95.1 ([Yang et al., 2019](https://arxiv.org/abs/1906.08237))         |
| Named Entity Recognition       | [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                 | F1     | **94.3**          | 93.3      | 93.5 ([Baevski et al., 2019](https://arxiv.org/abs/1903.07785))           |
| Cloze-style Question Answering | [ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)                         | EM/F1  | **90.6**/**91.2** | -         | 83.1/83.7 ([Li et al., 2019](https://www.aclweb.org/anthology/D19-6011/)) |
| Relation Classification        | [TACRED](https://nlp.stanford.edu/projects/tacred/)                          | F1     | **72.7**          | -         | 72.0 ([Wang et al. , 2020](https://arxiv.org/abs/2002.01808))             |
| Fine-grained Entity Typing     | [Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html) | F1     | **78.2**          | -         | 77.6 ([Wang et al. , 2020](https://arxiv.org/abs/2002.01808))             |

### Citation

If you find LUKE useful for your work, please cite the following paper:

```latex
@inproceedings{yamada2020luke,
  title={LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention},
  author={Ikuya Yamada and Akari Asai and Hiroyuki Shindo and Hideaki Takeda and Yuji Matsumoto},
  booktitle={EMNLP},
  year={2020}
}
```
