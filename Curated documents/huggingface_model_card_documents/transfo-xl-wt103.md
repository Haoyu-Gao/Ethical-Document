---
language: en
tags:
- text-generation
datasets:
- wikitext-103
task:
  name: Text Generation
  type: text-generation
model-index:
- name: transfo-xl-wt103
  results: []
---


# Transfo-xl-wt103

## Table of Contents
- [Model Details](#model-details)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation Information](#citation-information)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)


## Model Details
**Model Description:**
The Transformer-XL model is a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can reuse previously computed hidden-states to attend to longer context (memory). This model also uses adaptive softmax inputs and outputs (tied).
- **Developed by:** [Zihang Dai](dzihang@cs.cmu.edu), [Zhilin Yang](zhiliny@cs.cmu.edu), [Yiming Yang1](yiming@cs.cmu.edu), [Jaime Carbonell](jgc@cs.cmu.edu), [Quoc V. Le](qvl@google.com), [Ruslan Salakhutdinov](rsalakhu@cs.cmu.edu)
- **Shared by:** HuggingFace team
- **Model Type:** Text Generation
- **Language(s):** English
- **License:** [More information needed]
- **Resources for more information:**
  - [Research Paper](https://arxiv.org/pdf/1901.02860.pdf)
  - [GitHub Repo](https://github.com/kimiyoung/transformer-xl)
  - [HuggingFace Documentation](https://huggingface.co/docs/transformers/model_doc/transfo-xl#transformers.TransfoXLModel)


## Uses

#### Direct Use

This model can be used for text generation.
The authors provide additionally notes about the vocabulary used, in the [associated paper](https://arxiv.org/pdf/1901.02860.pdf): 

> We envision interesting applications of Transformer-XL in the fields of text generation, unsupervised feature learning, image and speech modeling.

#### Misuse and Out-of-scope Use
The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Risks, Limitations and Biases
**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).


## Training


#### Training Data

The authors provide additionally notes about the vocabulary used, in the [associated paper](https://arxiv.org/pdf/1901.02860.pdf): 

> best model trained the Wikitext-103 dataset. We seed the our Transformer-XL with a context of at most 512 consecutive tokens randomly sampled from the test set of Wikitext-103. Then, we run Transformer-XL to generate a pre-defined number of tokens (500 or 1,000 in our case). For each generation step, we first find the top-40 probabilities of the next-step distribution and sample from top-40 tokens based on the re-normalized distribution. To help reading, we detokenize the context, the generated text and the reference text.

The authors use the following pretraining corpora for the model, described in the [associated paper](https://arxiv.org/pdf/1901.02860.pdf):
- WikiText-103 (Merity et al., 2016), 


#### Training Procedure

##### Preprocessing
The authors provide additionally notes about the training procedure used, in the [associated paper](https://arxiv.org/pdf/1901.02860.pdf): 

> Similar to but different from enwik8, text8 con- tains 100M processed Wikipedia characters cre- ated by lowering case the text and removing any character other than the 26 letters a through z, and space. Due to the similarity, we simply adapt the best model and the same hyper-parameters on en- wik8 to text8 without further tuning.


## Evaluation

#### Results

| Method               | enwiki8  |text8 | One Billion Word | WT-103 | PTB (w/o finetuning) | 
|:--------------------:|---------:|:----:|:----------------:|:------:|:--------------------:|
| Transformer-XL.      | 0.99     | 1.08 | 21.8             | 18.3   |  54.5                |

## Citation Information

```bibtex

@misc{https://doi.org/10.48550/arxiv.1901.02860,
  doi = {10.48550/ARXIV.1901.02860},
  
  url = {https://arxiv.org/abs/1901.02860},
  
  author = {Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V. and Salakhutdinov, Ruslan},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context},
  
  publisher = {arXiv},
  
  year = {2019},
  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}


```

## How to Get Started With the Model
```
from transformers import TransfoXLTokenizer, TransfoXLModel
import torch

tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLModel.from_pretrained("transfo-xl-wt103")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

```









