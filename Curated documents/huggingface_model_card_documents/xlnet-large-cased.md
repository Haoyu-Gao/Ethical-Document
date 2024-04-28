---
language: en
license: mit
datasets:
- bookcorpus
- wikipedia
---

# XLNet (large-sized model) 

XLNet model pre-trained on English language. It was introduced in the paper [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Yang et al. and first released in [this repository](https://github.com/zihangdai/xlnet/). 

Disclaimer: The team releasing XLNet did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.

## Intended uses & limitations

The model is mostly intended to be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?search=xlnet) to look for fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation, you should look at models like GPT2.

## Usage

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-1906-08237,
  author    = {Zhilin Yang and
               Zihang Dai and
               Yiming Yang and
               Jaime G. Carbonell and
               Ruslan Salakhutdinov and
               Quoc V. Le},
  title     = {XLNet: Generalized Autoregressive Pretraining for Language Understanding},
  journal   = {CoRR},
  volume    = {abs/1906.08237},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.08237},
  eprinttype = {arXiv},
  eprint    = {1906.08237},
  timestamp = {Mon, 24 Jun 2019 17:28:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-08237.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
