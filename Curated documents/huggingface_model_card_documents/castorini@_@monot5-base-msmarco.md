---
{}
---
This model is a T5-base reranker fine-tuned on the MS MARCO passage dataset for 100k steps (or 10 epochs).

For better zero-shot performance (i.e., inference on other datasets), we recommend using `castorini/monot5-base-msmarco-10k`.

For more details on how to use it, check the following links:
- [A simple reranking example](https://github.com/castorini/pygaggle#a-simple-reranking-example)
- [Rerank MS MARCO passages](https://github.com/castorini/pygaggle/blob/master/docs/experiments-msmarco-passage-subset.md)
- [Rerank Robust04 documents](https://github.com/castorini/pygaggle/blob/master/docs/experiments-robust04-monot5-gpu.md)

Paper describing the model: [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://www.aclweb.org/anthology/2020.findings-emnlp.63/)