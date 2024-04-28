---
language: en
license: mit
tags:
- exbert
datasets:
- bookcorpus
- wikipedia
---

# Muppet: Massive Multi-task Representations with Pre-Finetuning
# RoBERTa large model

This is a Massive Multi-task Pre-finetuned version of Roberta large. It was introduced in
[this paper](https://arxiv.org/abs/2101.11038). The model improves over roberta-base in a wide range of GLUE, QA tasks (details can be found in the paper). The gains in
smaller datasets are significant. 

Note: This checkpoint does not contain the classificaiton/MRC heads used during pre-finetuning due to compatibility issues and hence you might get slightly lower performance than that reported in the paper on some datasets


## Model description

RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means
it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. 

More precisely, it was pretrained with the Masked language modeling (MLM) objective. Taking a sentence, the model
randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict
the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one
after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to
learn a bidirectional representation of the sentence.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard
classifier using the features produced by the BERT model as inputs.

## Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task.
See the [model hub](https://huggingface.co/models?filter=roberta) to look for fine-tuned versions on a task that
interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Model | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | SQuAD|
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:----:|
|  Roberta-large    | 90.2 | 92.2 | 94.7 | 96.4  | 63.6 | 91.2  | 90.9 | 88.1 | 88.7|
|  MUPPET Roberta-large    | 90.8 | 92.2 | 94.9 | 97.4  | - | -  | 91.4 | 92.8 | 89.4|

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2101-11038,
  author    = {Armen Aghajanyan and
               Anchit Gupta and
               Akshat Shrivastava and
               Xilun Chen and
               Luke Zettlemoyer and
               Sonal Gupta},
  title     = {Muppet: Massive Multi-task Representations with Pre-Finetuning},
  journal   = {CoRR},
  volume    = {abs/2101.11038},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.11038},
  archivePrefix = {arXiv},
  eprint    = {2101.11038},
  timestamp = {Sun, 31 Jan 2021 17:23:50 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2101-11038.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```