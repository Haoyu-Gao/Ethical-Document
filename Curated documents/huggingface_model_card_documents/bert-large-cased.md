---
language: en
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
---

# BERT large model (cased)

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://arxiv.org/abs/1810.04805) and first released in
[this repository](https://github.com/google-research/bert). This model is cased: it makes a difference
between english and English.

Disclaimer: The team releasing BERT did not write a model card for this model so this model card has been written by
the Hugging Face team.

## Model description

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it
was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes
  they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to
  predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard
classifier using the features produced by the BERT model as inputs.

This model has the following configuration:

- 24-layer
- 1024 hidden dimension
- 16 attention heads
- 336M parameters.

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=bert) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-large-cased')
>>> unmasker("Hello I'm a [MASK] model.")
[
   {
      "sequence":"[CLS] Hello I'm a male model. [SEP]",
      "score":0.22748498618602753,
      "token":2581,
      "token_str":"male"
   },
   {
      "sequence":"[CLS] Hello I'm a fashion model. [SEP]",
      "score":0.09146175533533096,
      "token":4633,
      "token_str":"fashion"
   },
   {
      "sequence":"[CLS] Hello I'm a new model. [SEP]",
      "score":0.05823173746466637,
      "token":1207,
      "token_str":"new"
   },
   {
      "sequence":"[CLS] Hello I'm a super model. [SEP]",
      "score":0.04488750174641609,
      "token":7688,
      "token_str":"super"
   },
   {
      "sequence":"[CLS] Hello I'm a famous model. [SEP]",
      "score":0.03271442651748657,
      "token":2505,
      "token_str":"famous"
   }
]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained("bert-large-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = TFBertModel.from_pretrained("bert-large-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

### Limitations and bias

Even if the training data used for this model could be characterized as fairly neutral, this model can have biased
predictions:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-large-cased')
>>> unmasker("The man worked as a [MASK].")
[
   {
      "sequence":"[CLS] The man worked as a doctor. [SEP]",
      "score":0.0645911768078804,
      "token":3995,
      "token_str":"doctor"
   },
   {
      "sequence":"[CLS] The man worked as a cop. [SEP]",
      "score":0.057450827211141586,
      "token":9947,
      "token_str":"cop"
   },
   {
      "sequence":"[CLS] The man worked as a mechanic. [SEP]",
      "score":0.04392256215214729,
      "token":19459,
      "token_str":"mechanic"
   },
   {
      "sequence":"[CLS] The man worked as a waiter. [SEP]",
      "score":0.03755280375480652,
      "token":17989,
      "token_str":"waiter"
   },
   {
      "sequence":"[CLS] The man worked as a teacher. [SEP]",
      "score":0.03458863124251366,
      "token":3218,
      "token_str":"teacher"
   }
]

>>> unmasker("The woman worked as a [MASK].")
[
   {
      "sequence":"[CLS] The woman worked as a nurse. [SEP]",
      "score":0.2572779953479767,
      "token":7439,
      "token_str":"nurse"
   },
   {
      "sequence":"[CLS] The woman worked as a waitress. [SEP]",
      "score":0.16706500947475433,
      "token":15098,
      "token_str":"waitress"
   },
   {
      "sequence":"[CLS] The woman worked as a teacher. [SEP]",
      "score":0.04587847739458084,
      "token":3218,
      "token_str":"teacher"
   },
   {
      "sequence":"[CLS] The woman worked as a secretary. [SEP]",
      "score":0.03577028587460518,
      "token":4848,
      "token_str":"secretary"
   },
   {
      "sequence":"[CLS] The woman worked as a maid. [SEP]",
      "score":0.03298963978886604,
      "token":13487,
      "token_str":"maid"
   }
]
```

This bias will also affect all fine-tuned versions of this model.

## Training data

The BERT model was pretrained on [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038
unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and
headers).

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

With probability 0.5, sentence A and sentence B correspond to two consecutive sentences in the original corpus and in
the other cases, it's another random sentence in the corpus. Note that what is considered a sentence here is a
consecutive span of text usually longer than a single sentence. The only constrain is that the result with the two
"sentences" has a combined length of less than 512 tokens.

The details of the masking procedure for each sentence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

### Pretraining

The model was trained on 4 cloud TPUs in Pod configuration (16 TPU chips total) for one million steps with a batch size
of 256. The sequence length was limited to 128 tokens for 90% of the steps and 512 for the remaining 10%. The optimizer
used is Adam with a learning rate of 1e-4, \\(\beta_{1} = 0.9\\) and \\(\beta_{2} = 0.999\\), a weight decay of 0.01,
learning rate warmup for 10,000 steps and linear decay of the learning rate after.

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Model                                    | SQUAD 1.1 F1/EM | Multi NLI Accuracy
---------------------------------------- | :-------------: | :----------------:
BERT-Large, Cased (Original)             | 91.5/84.8       | 86.09

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
