---
language: en
license: apache-2.0
datasets:
- squad
metrics:
- squad
model-index:
- name: distilbert-base-cased-distilled-squad
  results:
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squad
      type: squad
      config: plain_text
      split: validation
    metrics:
    - type: exact_match
      value: 79.5998
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTViZDA2Y2E2NjUyMjNjYjkzNTUzODc5OTk2OTNkYjQxMDRmMDhlYjdmYWJjYWQ2N2RlNzY1YmI3OWY1NmRhOSIsInZlcnNpb24iOjF9.ZJHhboAMwsi3pqU-B-XKRCYP_tzpCRb8pEjGr2Oc-TteZeoWHI8CXcpDxugfC3f7d_oBcKWLzh3CClQxBW1iAQ
    - type: f1
      value: 86.9965
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZWZlMzY2MmE1NDNhOGNjNWRmODg0YjQ2Zjk5MjUzZDQ2MDYxOTBlMTNhNzQ4NTA2NjRmNDU3MGIzMTYwMmUyOSIsInZlcnNpb24iOjF9.z0ZDir87aT7UEmUeDm8Uw0oUdAqzlBz343gwnsQP3YLfGsaHe-jGlhco0Z7ISUd9NokyCiJCRc4NNxJQ83IuCw
---

# DistilBERT base cased distilled SQuAD

## Table of Contents
- [Model Details](#model-details)
- [How To Get Started With the Model](#how-to-get-started-with-the-model)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation Information](#citation-information)
- [Model Card Authors](#model-card-authors)

## Model Details

**Model Description:** The DistilBERT model was proposed in the blog post [Smaller, faster, cheaper, lighter: Introducing DistilBERT, adistilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5), and the paper [DistilBERT, adistilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than *bert-base-uncased*, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.

This model is a fine-tune checkpoint of [DistilBERT-base-cased](https://huggingface.co/distilbert-base-cased), fine-tuned using (a second step of) knowledge distillation on [SQuAD v1.1](https://huggingface.co/datasets/squad). 

- **Developed by:** Hugging Face
- **Model Type:** Transformer-based language model
- **Language(s):** English 
- **License:** Apache 2.0
- **Related Models:** [DistilBERT-base-cased](https://huggingface.co/distilbert-base-cased)
- **Resources for more information:**
  - See [this repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) for more about Distil\* (a class of compressed models including this model)
  - See [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108) for more information about knowledge distillation and the training procedure

## How to Get Started with the Model 

Use the code below to get started with the model. 

```python
>>> from transformers import pipeline
>>> question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

>>> context = r"""
... Extractive Question Answering is the task of extracting an answer from a text given a question. An example     of a
... question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
... a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
... """

>>> result = question_answerer(question="What is a good example of a question answering dataset?",     context=context)
>>> print(
... f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
...)

Answer: 'SQuAD dataset', score: 0.5152, start: 147, end: 160
```

Here is how to use this model in PyTorch:

```python
from transformers import DistilBertTokenizer, DistilBertModel
import torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = DistilBertModel.from_pretrained('distilbert-base-cased-distilled-squad')

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print(outputs)
```

And in TensorFlow: 

```python
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="tf")
outputs = model(**inputs)

answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)
```

## Uses

This model can be used for question answering.

#### Misuse and Out-of-scope Use

The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Risks, Limitations and Biases

**CONTENT WARNING: Readers should be aware that language generated by this model can be disturbing or offensive to some and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). Predictions generated by the model can include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups. For example:


```python
>>> from transformers import pipeline
>>> question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

>>> context = r"""
... Alice is sitting on the bench. Bob is sitting next to her.
... """

>>> result = question_answerer(question="Who is the CEO?", context=context)
>>> print(
... f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
...)

Answer: 'Bob', score: 0.7527, start: 32, end: 35
```

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## Training

#### Training Data

The [distilbert-base-cased model](https://huggingface.co/distilbert-base-cased) was trained using the same data as the [distilbert-base-uncased model](https://huggingface.co/distilbert-base-uncased). The [distilbert-base-uncased model](https://huggingface.co/distilbert-base-uncased) model describes it's training data as: 

> DistilBERT pretrained on the same data as BERT, which is [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038 unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and headers).

To learn more about the SQuAD v1.1 dataset, see the [SQuAD v1.1 data card](https://huggingface.co/datasets/squad).

#### Training Procedure

##### Preprocessing

See the [distilbert-base-cased model card](https://huggingface.co/distilbert-base-cased) for further details.

##### Pretraining

See the [distilbert-base-cased model card](https://huggingface.co/distilbert-base-cased) for further details. 

## Evaluation

As discussed in the [model repository](https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/README.md)

> This model reaches a F1 score of 87.1 on the [SQuAD v1.1] dev set (for comparison, BERT bert-base-cased version reaches a F1 score of 88.7).	

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). We present the hardware type and hours used based on the [associated paper](https://arxiv.org/pdf/1910.01108.pdf). Note that these details are just for training DistilBERT, not including the fine-tuning with SQuAD.

- **Hardware Type:** 8 16GB V100 GPUs
- **Hours used:** 90 hours
- **Cloud Provider:** Unknown
- **Compute Region:** Unknown
- **Carbon Emitted:** Unknown

## Technical Specifications

See the [associated paper](https://arxiv.org/abs/1910.01108) for details on the modeling architecture, objective, compute infrastructure, and training details.

## Citation Information

```bibtex
@inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMC^2 Workshop},
  year={2019}
}
```

APA: 
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

## Model Card Authors

This model card was written by the Hugging Face team. 
