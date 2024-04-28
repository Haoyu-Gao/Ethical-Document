---
language:
- multilingual
- en
- de
---

# xlm-clm-ende-1024

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Environmental Impact](#environmental-impact)
7. [Technical Specifications](#technical-specifications)
8. [Citation](#citation)
9. [Model Card Authors](#model-card-authors)
10. [How To Get Started With the Model](#how-to-get-started-with-the-model)


# Model Details

The XLM model was proposed in [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample, Alexis Conneau. xlm-clm-ende-1024 is a transformer pretrained using a causal language modeling (CLM) objective (next token prediction) for English-German.

## Model Description

- **Developed by:** Guillaume Lample, Alexis Conneau, see [associated paper](https://arxiv.org/abs/1901.07291)
- **Model type:** Language model
- **Language(s) (NLP):** English-German
- **License:** Unknown
- **Related Models:** [xlm-clm-enfr-1024](https://huggingface.co/xlm-clm-enfr-1024), [xlm-mlm-ende-1024](https://huggingface.co/xlm-mlm-ende-1024), [xlm-mlm-enfr-1024](https://huggingface.co/xlm-mlm-enfr-1024), [xlm-mlm-enro-1024](https://huggingface.co/xlm-mlm-enro-1024)
- **Resources for more information:** 
  - [Associated paper](https://arxiv.org/abs/1901.07291)
  - [GitHub Repo](https://github.com/facebookresearch/XLM)
  - [Hugging Face Multilingual Models for Inference docs](https://huggingface.co/docs/transformers/v4.20.1/en/multilingual#xlm-with-language-embeddings)

# Uses

## Direct Use

The model is a language model. The model can be used for causal language modeling. 

## Downstream Use

To learn more about this task and potential downstream uses, see the [Hugging Face Multilingual Models for Inference](https://huggingface.co/docs/transformers/v4.20.1/en/multilingual#xlm-with-language-embeddings) docs.

## Out-of-Scope Use

The model should not be used to intentionally create hostile or alienating environments for people. 

# Bias, Risks, and Limitations

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).

## Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

# Training

See the [associated paper](https://arxiv.org/pdf/1901.07291.pdf) for details on the training data and training procedure.
 
# Evaluation

## Testing Data, Factors & Metrics

See the [associated paper](https://arxiv.org/pdf/1901.07291.pdf) for details on the testing data, factors and metrics.

## Results 

For xlm-clm-ende-1024 results, see Table 2 of the [associated paper](https://arxiv.org/pdf/1901.07291.pdf).

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** More information needed
- **Hours used:** More information needed
- **Cloud Provider:** More information needed
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Technical Specifications

The model developers write: 

> We implement all our models in PyTorch (Paszke et al., 2017), and train them on 64 Volta GPUs for the language modeling tasks, and 8 GPUs for the MT tasks. We use float16 operations to speed up training and to reduce the memory usage of our models.

See the [associated paper](https://arxiv.org/pdf/1901.07291.pdf) for further details.

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

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

```python
import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel

tokenizer = XLMTokenizer.from_pretrained("xlm-clm-ende-1024")
model = XLMWithLMHeadModel.from_pretrained("xlm-clm-ende-1024")

input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1

language_id = tokenizer.lang2id["en"]  # 0
langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

# We reshape it to be of size (batch_size, sequence_length)
langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)

outputs = model(input_ids, langs=langs)
```

</details>