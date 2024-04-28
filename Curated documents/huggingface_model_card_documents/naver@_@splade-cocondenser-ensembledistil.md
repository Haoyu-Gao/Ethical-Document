---
language: en
license: cc-by-nc-sa-4.0
tags:
- splade
- query-expansion
- document-expansion
- bag-of-words
- passage-retrieval
- knowledge-distillation
datasets:
- ms_marco
---

## SPLADE CoCondenser EnsembleDistil

SPLADE model for passage retrieval. For additional details, please visit:
* paper: https://arxiv.org/abs/2205.04733
* code: https://github.com/naver/splade

| | MRR@10 (MS MARCO dev) | R@1000 (MS MARCO dev) |
| --- | --- | --- |
| `splade-cocondenser-ensembledistil` | 38.3 | 98.3 |

## Citation

If you use our checkpoint, please cite our work:

```
@misc{https://doi.org/10.48550/arxiv.2205.04733,
  doi = {10.48550/ARXIV.2205.04733},
  url = {https://arxiv.org/abs/2205.04733},
  author = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, Stéphane},
  keywords = {Information Retrieval (cs.IR), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```