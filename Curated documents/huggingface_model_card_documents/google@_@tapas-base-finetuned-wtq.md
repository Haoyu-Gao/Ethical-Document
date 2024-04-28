---
language: en
license: apache-2.0
tags:
- tapas
datasets:
- wikitablequestions
---

# TAPAS base model fine-tuned on WikiTable Questions (WTQ)

This model has 2 versions which can be used. The default version corresponds to the `tapas_wtq_wikisql_sqa_inter_masklm_base_reset` checkpoint of the [original Github repository](https://github.com/google-research/tapas).
This model was pre-trained on MLM and an additional step which the authors call intermediate pre-training, and then fine-tuned in a chain on [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253), [WikiSQL](https://github.com/salesforce/WikiSQL) and finally [WTQ](https://github.com/ppasupat/WikiTableQuestions). It uses relative position embeddings (i.e. resetting the position index at every cell of the table).

The other (non-default) version which can be used is: 
- `no_reset`, which corresponds to `tapas_wtq_wikisql_sqa_inter_masklm_base` (intermediate pre-training, absolute position embeddings). 

Disclaimer: The team releasing TAPAS did not write a model card for this model so this model card has been written by
the Hugging Face team and contributors.

## Results

Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.5062 | [tapas-large-finetuned-wtq (with absolute pos embeddings)](https://huggingface.co/google/tapas-large-finetuned-wtq/tree/no_reset)
LARGE | reset | 0.5097 | [tapas-large-finetuned-wtq](https://huggingface.co/google/tapas-large-finetuned-wtq/tree/main)
**BASE** | **noreset** | **0.4525** | [tapas-base-finetuned-wtq (with absolute pos embeddings)](https://huggingface.co/google/tapas-base-finetuned-wtq/tree/no_reset)
**BASE** | **reset** | **0.4638** | [tapas-base-finetuned-wtq](https://huggingface.co/google/tapas-base-finetuned-wtq/tree/main)
MEDIUM | noreset | 0.4324 | [tapas-medium-finetuned-wtq (with absolute pos embeddings)](https://huggingface.co/google/tapas-medium-finetuned-wtq/tree/no_reset)
MEDIUM | reset | 0.4324 | [tapas-medium-finetuned-wtq](https://huggingface.co/google/tapas-medium-finetuned-wtq/tree/main)
SMALL | noreset | 0.3681 | [tapas-small-finetuned-wtq (with absolute pos embeddings)](https://huggingface.co/google/tapas-small-finetuned-wtq/tree/no_reset)
SMALL | reset | 0.3762 | [tapas-small-finetuned-wtq](https://huggingface.co/google/tapas-small-finetuned-wtq/tree/main)
MINI | noreset | 0.2783 | [tapas-mini-finetuned-wtq (with absolute pos embeddings)](https://huggingface.co/google/tapas-mini-finetuned-wtq/tree/no_reset)
MINI | reset | 0.2854 | [tapas-mini-finetuned-wtq](https://huggingface.co/google/tapas-mini-finetuned-wtq/tree/main)
TINY | noreset | 0.0823 | [tapas-tiny-finetuned-wtq (with absolute pos embeddings)](https://huggingface.co/google/tapas-tiny-finetuned-wtq/tree/no_reset)
TINY | reset | 0.1039 | [tapas-tiny-finetuned-wtq](https://huggingface.co/google/tapas-tiny-finetuned-wtq/tree/main)

## Model description

TAPAS is a BERT-like transformers model pretrained on a large corpus of English data from Wikipedia in a self-supervised fashion. 
This means it was pretrained on the raw tables and associated texts only, with no humans labelling them in any way (which is why it
can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a (flattened) table and associated context, the model randomly masks 15% of the words in 
  the input, then runs the entire (partially masked) sequence through the model. The model then has to predict the masked words. 
  This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, 
  or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional 
  representation of a table and associated text.
- Intermediate pre-training: to encourage numerical reasoning on tables, the authors additionally pre-trained the model by creating 
  a balanced dataset of millions of syntactically created training examples. Here, the model must predict (classify) whether a sentence 
  is supported or refuted by the contents of a table. The training examples are created based on synthetic as well as counterfactual statements.

This way, the model learns an inner representation of the English language used in tables and associated texts, which can then be used 
to extract features useful for downstream tasks such as answering questions about a table, or determining whether a sentence is entailed
or refuted by the contents of a table. Fine-tuning is done by adding a cell selection head and aggregation head on top of the pre-trained model, and then jointly train these randomly initialized classification heads with the base model on SQa, WikiSQL and finally WTQ. 


## Intended uses & limitations

You can use this model for answering questions related to a table.

For code examples, we refer to the documentation of TAPAS on the HuggingFace website. 


## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

```
[CLS] Question [SEP] Flattened table [SEP]
```

The authors did first convert the WTQ dataset into the format of SQA using automatic conversion scripts.

### Fine-tuning

The model was fine-tuned on 32 Cloud TPU v3 cores for 50,000 steps with maximum sequence length 512 and batch size of 512.
In this setup, fine-tuning takes around 10 hours. The optimizer used is Adam with a learning rate of 1.93581e-5, and a warmup 
ratio of 0.128960. An inductive bias is added such that the model only selects cells of the same column. This is reflected by the 
`select_one_column` parameter of `TapasConfig`. See the [paper](https://arxiv.org/abs/2004.02349) for more details (tables 11 and
12). 


### BibTeX entry and citation info

```bibtex
@misc{herzig2020tapas,
      title={TAPAS: Weakly Supervised Table Parsing via Pre-training}, 
      author={Jonathan Herzig and Paweł Krzysztof Nowak and Thomas Müller and Francesco Piccinno and Julian Martin Eisenschlos},
      year={2020},
      eprint={2004.02349},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

```bibtex
@misc{eisenschlos2020understanding,
      title={Understanding tables with intermediate pre-training}, 
      author={Julian Martin Eisenschlos and Syrine Krichene and Thomas Müller},
      year={2020},
      eprint={2010.00571},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@article{DBLP:journals/corr/PasupatL15,
  author    = {Panupong Pasupat and
               Percy Liang},
  title     = {Compositional Semantic Parsing on Semi-Structured Tables},
  journal   = {CoRR},
  volume    = {abs/1508.00305},
  year      = {2015},
  url       = {http://arxiv.org/abs/1508.00305},
  archivePrefix = {arXiv},
  eprint    = {1508.00305},
  timestamp = {Mon, 13 Aug 2018 16:47:37 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/PasupatL15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```