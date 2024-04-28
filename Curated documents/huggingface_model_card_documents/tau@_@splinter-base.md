---
language: en
license: apache-2.0
tags:
- splinter
- SplinterModel
---

# Splinter base model  

Splinter-base is the pretrained model discussed in the paper [Few-Shot Question Answering by Pretraining Span Selection](https://aclanthology.org/2021.acl-long.239/) (at ACL 2021). Its original repository can be found [here](https://github.com/oriram/splinter). The model is case-sensitive.

Note: This model **doesn't** contain the pretrained weights for the QASS layer (see paper for details), and therefore the QASS layer is randomly initialized upon loading it. For the model **with** those weights, see [tau/splinter-base-qass](https://huggingface.co/tau/splinter-base-qass).

## Model description

Splinter is a model that is pretrained in a self-supervised fashion for few-shot question answering. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. 

More precisely, it was pretrained with the Recurring Span Selection (RSS) objective, which emulates the span selection process involved in extractive question answering. Given a text, clusters of recurring spans (n-grams that appear more than once in the text) are first identified. For each such cluster, all of its instances but one are replaced with a special `[QUESTION]` token, and the model should select the correct (i.e., unmasked) span for each masked one. The model also defines the Question-Aware Span selection (QASS) layer, which selects spans conditioned on a specific question (in order to perform multiple predictions).


## Intended uses & limitations

The prime use for this model is few-shot extractive QA.

## Pretraining

The model was pretrained on a v3-8 TPU for 2.4M steps. The training data is based on **Wikipedia** and **BookCorpus**. See the paper for more details.

### BibTeX entry and citation info

```bibtex
@inproceedings{ram-etal-2021-shot,
    title = "Few-Shot Question Answering by Pretraining Span Selection",
    author = "Ram, Ori  and
      Kirstain, Yuval  and
      Berant, Jonathan  and
      Globerson, Amir  and
      Levy, Omer",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.239",
    doi = "10.18653/v1/2021.acl-long.239",
    pages = "3066--3079",
}
```