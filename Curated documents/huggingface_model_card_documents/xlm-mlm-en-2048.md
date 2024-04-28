---
language: en
license: cc-by-nc-4.0
tags:
- exbert
---

# xlm-mlm-en-2048

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Environmental Impact](#environmental-impact)
7. [Citation](#citation)
8. [Model Card Authors](#model-card-authors)
9. [How To Get Started With the Model](#how-to-get-started-with-the-model)


# Model Details

The XLM model was proposed in [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau. It’s a transformer pretrained with either a causal language modeling (CLM) objective (next token prediction), a masked language modeling (MLM) objective (BERT-like), or
a Translation Language Modeling (TLM) object (extension of BERT’s MLM to multiple language inputs). This model is trained with a masked language modeling objective on English text. 

## Model Description 

- **Developed by:** Researchers affiliated with Facebook AI, see [associated paper](https://arxiv.org/abs/1901.07291) and [GitHub Repo](https://github.com/facebookresearch/XLM)
- **Model type:** Language model
- **Language(s) (NLP):** English
- **License:** CC-BY-NC-4.0
- **Related Models:** Other [XLM models](https://huggingface.co/models?sort=downloads&search=xlm)
- **Resources for more information:** 
  - [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau (2019)
  - [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf) by Conneau et al. (2020)
  - [GitHub Repo](https://github.com/facebookresearch/XLM)
  - [Hugging Face XLM docs](https://huggingface.co/docs/transformers/model_doc/xlm)

# Uses

## Direct Use

The model is a language model. The model can be used for masked language modeling. 

## Downstream Use

To learn more about this task and potential downstream uses, see the Hugging Face [fill mask docs](https://huggingface.co/tasks/fill-mask) and the [Hugging Face Multilingual Models for Inference](https://huggingface.co/docs/transformers/v4.20.1/en/multilingual#xlm-with-language-embeddings) docs. Also see the [associated paper](https://arxiv.org/abs/1901.07291).

## Out-of-Scope Use

The model should not be used to intentionally create hostile or alienating environments for people. 

# Bias, Risks, and Limitations

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).

## Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

# Training 

More information needed. See the [associated GitHub Repo](https://github.com/facebookresearch/XLM).
 
# Evaluation 

More information needed. See the [associated GitHub Repo](https://github.com/facebookresearch/XLM).

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** More information needed
- **Hours used:** More information needed
- **Cloud Provider:** More information needed
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Citation 

**BibTeX:**

```bibtex
@article{lample2019cross,
  title={Cross-lingual language model pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={arXiv preprint arXiv:1901.07291},
  year={2019}
}
```

**APA:**
- Lample, G., & Conneau, A. (2019). Cross-lingual language model pretraining. arXiv preprint arXiv:1901.07291.

# Model Card Authors 

This model card was written by the team at Hugging Face.

# How to Get Started with the Model

Use the code below to get started with the model. See the [Hugging Face XLM docs](https://huggingface.co/docs/transformers/model_doc/xlm) for more examples.

```python
from transformers import XLMTokenizer, XLMModel
import torch

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")
model = XLMModel.from_pretrained("xlm-mlm-en-2048")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

<a href="https://huggingface.co/exbert/?model=xlm-mlm-en-2048">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
