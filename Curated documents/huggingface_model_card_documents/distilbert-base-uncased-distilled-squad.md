---
language: en
license: apache-2.0
datasets:
- squad
widget:
- text: Which name is also used to describe the Amazon rainforest in English?
  context: 'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish:
    Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch:
    Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is
    a moist broadleaf forest that covers most of the Amazon basin of South America.
    This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which
    5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This
    region includes territory belonging to nine nations. The majority of the forest
    is contained within Brazil, with 60% of the rainforest, followed by Peru with
    13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia,
    Guyana, Suriname and French Guiana. States or departments in four nations contain
    "Amazonas" in their names. The Amazon represents over half of the planet''s remaining
    rainforests, and comprises the largest and most biodiverse tract of tropical rainforest
    in the world, with an estimated 390 billion individual trees divided into 16,000
    species.'
- text: How many square kilometers of rainforest is covered in the basin?
  context: 'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish:
    Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch:
    Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is
    a moist broadleaf forest that covers most of the Amazon basin of South America.
    This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which
    5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This
    region includes territory belonging to nine nations. The majority of the forest
    is contained within Brazil, with 60% of the rainforest, followed by Peru with
    13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia,
    Guyana, Suriname and French Guiana. States or departments in four nations contain
    "Amazonas" in their names. The Amazon represents over half of the planet''s remaining
    rainforests, and comprises the largest and most biodiverse tract of tropical rainforest
    in the world, with an estimated 390 billion individual trees divided into 16,000
    species.'
---

# DistilBERT base uncased distilled SQuAD

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

This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned using (a second step of) knowledge distillation on [SQuAD v1.1](https://huggingface.co/datasets/squad). 

- **Developed by:** Hugging Face
- **Model Type:** Transformer-based language model
- **Language(s):** English 
- **License:** Apache 2.0
- **Related Models:** [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased)
- **Resources for more information:**
  - See [this repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) for more about Distil\* (a class of compressed models including this model)
  - See [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108) for more information about knowledge distillation and the training procedure

## How to Get Started with the Model 

Use the code below to get started with the model. 

```python
>>> from transformers import pipeline
>>> question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

>>> context = r"""
... Extractive Question Answering is the task of extracting an answer from a text given a question. An example     of a
... question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
... a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
... """

>>> result = question_answerer(question="What is a good example of a question answering dataset?",     context=context)
>>> print(
... f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
...)

Answer: 'SQuAD dataset', score: 0.4704, start: 147, end: 160
```

Here is how to use this model in PyTorch:

```python
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = torch.argmax(outputs.start_logits)
answer_end_index = torch.argmax(outputs.end_logits)

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)
```

And in TensorFlow: 

```python
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

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
>>> question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

>>> context = r"""
... Alice is sitting on the bench. Bob is sitting next to her.
... """

>>> result = question_answerer(question="Who is the CEO?", context=context)
>>> print(
... f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
...)

Answer: 'Bob', score: 0.4183, start: 32, end: 35
```

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## Training

#### Training Data

The [distilbert-base-uncased model](https://huggingface.co/distilbert-base-uncased) model describes it's training data as: 

> DistilBERT pretrained on the same data as BERT, which is [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038 unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and headers).

To learn more about the SQuAD v1.1 dataset, see the [SQuAD v1.1 data card](https://huggingface.co/datasets/squad).

#### Training Procedure

##### Preprocessing

See the [distilbert-base-uncased model card](https://huggingface.co/distilbert-base-uncased) for further details.

##### Pretraining

See the [distilbert-base-uncased model card](https://huggingface.co/distilbert-base-uncased) for further details. 

## Evaluation

As discussed in the [model repository](https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/README.md)

> This model reaches a F1 score of 86.9 on the [SQuAD v1.1] dev set (for comparison, Bert bert-base-uncased version reaches a F1 score of 88.5).

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
