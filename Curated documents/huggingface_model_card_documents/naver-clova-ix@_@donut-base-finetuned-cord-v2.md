---
license: mit
tags:
- donut
- image-to-text
- vision
---

# Donut (base-sized model, fine-tuned on CORD) 

Donut model fine-tuned on CORD. It was introduced in the paper [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) by Geewok et al. and first released in [this repository](https://github.com/clovaai/donut).

Disclaimer: The team releasing Donut did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

Donut consists of a vision encoder (Swin Transformer) and a text decoder (BART). Given an image, the encoder first encodes the image into a tensor of embeddings (of shape batch_size, seq_len, hidden_size), after which the decoder autoregressively generates text, conditioned on the encoding of the encoder. 

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg)

## Intended uses & limitations

This model is fine-tuned on CORD, a document parsing dataset.

We refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/donut) which includes code examples.

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2111-15664,
  author    = {Geewook Kim and
               Teakgyu Hong and
               Moonbin Yim and
               Jinyoung Park and
               Jinyeong Yim and
               Wonseok Hwang and
               Sangdoo Yun and
               Dongyoon Han and
               Seunghyun Park},
  title     = {Donut: Document Understanding Transformer without {OCR}},
  journal   = {CoRR},
  volume    = {abs/2111.15664},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.15664},
  eprinttype = {arXiv},
  eprint    = {2111.15664},
  timestamp = {Thu, 02 Dec 2021 10:50:44 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-15664.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```