---
language: en
---


# OLM RoBERTa/BERT December 2022

This is a more up-to-date version of the [original BERT](https://huggingface.co/bert-base-cased) and [original RoBERTa](https://huggingface.co/roberta-base).
In addition to being more up-to-date, it also tends to perform better than the original BERT on standard benchmarks.
We think it is fair to directly compare our model to the original BERT because our model was trained with about the same level of compute as the original BERT, and the architecture of BERT and RoBERTa are basically the same.
The original RoBERTa takes an order of magnitude more compute, although our model is also not that different in performance from the original RoBERTa on many standard benchmarks.
Our model was trained on a cleaned December 2022 snapshot of Common Crawl and Wikipedia.

This model was created as part of the OLM project, which has the goal of continuously training and releasing models that are up-to-date and comparable in standard language model performance to their static counterparts.
This is important because we want our models to know about events like COVID or 
a presidential election right after they happen.

## Intended uses

You can use the raw model for masked language modeling, but it's mostly intended to
be fine-tuned on a downstream task, such as sequence classification, token classification or question answering. 

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='olm/olm-roberta-base-dec-2022')
>>> unmasker("Hello I'm a <mask> model.")
[{'score': 0.04252663999795914,
  'token': 631,
  'token_str': ' new',
  'sequence': "Hello I'm a new model."},
 {'score': 0.034064881503582,
  'token': 4750,
  'token_str': ' female',
  'sequence': "Hello I'm a female model."},
 {'score': 0.03066524863243103,
 'token': 932,
 'token_str': ' business',
 'sequence': "Hello I'm a business model."},
 {'score': 0.029599128291010857,
  'token': 10345,
  'token_str': ' junior',
  'sequence': "Hello I'm a junior model."},
 {'score': 0.025790784507989883,
  'token': 2219,
  'token_str': ' human',
  'sequence': "Hello I'm a human model."}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import AutoTokenizer, RobertaModel
tokenizer = AutoTokenizer.from_pretrained('olm/olm-roberta-base-dec-2022')
model = RobertaModel.from_pretrained("olm/olm-roberta-base-dec-2022")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

## Dataset

The model and tokenizer were trained with this [December 2022 cleaned Common Crawl dataset](https://huggingface.co/datasets/olm/olm-CC-MAIN-2022-49-sampling-ratio-olm-0.15114822547) plus this [December 2022 cleaned Wikipedia dataset](https://huggingface.co/datasets/olm/olm-wikipedia-20221220).\
The tokenized version of these concatenated datasets is [here](https://huggingface.co/datasets/olm/olm-december-2022-tokenized-512).\
The datasets were created with this [repo](https://github.com/huggingface/olm-datasets).

## Training

The model was trained according to the OLM BERT/RoBERTa instructions at this [repo](https://github.com/huggingface/olm-training).

## Evaluation results

The model achieves the following results after tuning on GLUE tasks:

| Task | Metric   | Original BERT   | OLM RoBERTa Dec 2022 (Ours) |
|:-----|:---------|----------------:|----------------------------:|
|cola  |mcc       |**0.5889**       |0.28067                      |
|sst2  |acc       |0.9181           |**0.9275**                   |
|mrpc  |acc/f1    |**0.9182**/0.8923|0.8662/**0.9033**            |
|stsb  |pear/spear|0.8822/0.8794    |**0.8870**/**0.8857**        |
|qqp   |acc/f1    |0.9071/0.8748    |**0.9097**/**0.8791**        |
|mnli  |acc/acc_mm|0.8400/0.8410    |**0.8576**/**0.8621**        |
|qnli  |acc       |0.9075           |**0.9192**                   |
|rte   |acc       |0.6296           |**0.6390**                   |
|wnli  |acc       |0.4000           |**0.4648**                   |

For both the original BERT and our model, we used the Hugging Face run_glue.py script [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
For both models, we used the default fine-tuning hyperparameters and we averaged the results over five training seeds. These are the results for the GLUE dev sets, which can be a bit different than the results for the test sets.