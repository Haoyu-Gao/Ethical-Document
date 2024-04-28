---
language:
- en
license: apache-2.0
datasets:
- allenai/scirepeval
---

<!-- Provide a quick summary of what the model is/does. -->

## SPECTER2

<!-- Provide a quick summary of what the model is/does. -->

SPECTER2 is the successor to [SPECTER](https://huggingface.co/allenai/specter) and is capable of generating task specific embeddings for scientific tasks when paired with [adapters](https://huggingface.co/models?search=allenai/specter-2_).
This is the base model to be used along with the adapters.
Given the combination of title and abstract of a scientific paper or a short texual query, the model can be used to generate effective embeddings to be used in downstream applications.

**Note:For general embedding purposes, please use [allenai/specter2](https://huggingface.co/allenai/specter2).**

**To get the best performance on a downstream task type please load the associated adapter with the base model as in the example below.**

**Dec 2023 Update:**

Model usage updated to be compatible with latest versions of transformers and adapters (newly released update to adapter-transformers) libraries.

**Aug 2023 Update:**
1. **The SPECTER2 Base and proximity adapter models have been renamed in Hugging Face based upon usage patterns as follows:**

|Old Name|New Name|
|--|--|
|allenai/specter2|[allenai/specter2_base](https://huggingface.co/allenai/specter2_base)|
|allenai/specter2_proximity|[allenai/specter2](https://huggingface.co/allenai/specter2)|

2. **We have a parallel version (termed [aug2023refresh](https://huggingface.co/allenai/specter2_aug2023refresh)) where the base transformer encoder version is pre-trained on a collection of newer papers (published after 2018).
   However, for benchmarking purposes, please continue using the current version.**


An [adapter](https://adapterhub.ml) for the [allenai/specter2_base](https://huggingface.co/allenai/specter2_base) model that was trained on the [allenai/scirepeval](https://huggingface.co/datasets/allenai/scirepeval/) dataset.

This adapter was created for usage with the **[adapters](https://github.com/adapter-hub/adapters)** library.

# Model Details

## Model Description

SPECTER2 has been trained on over 6M triplets of scientific paper citations, which are available [here](https://huggingface.co/datasets/allenai/scirepeval/viewer/cite_prediction_new/evaluation).
Post that it is trained with additionally attached task format specific adapter modules on all the [SciRepEval](https://huggingface.co/datasets/allenai/scirepeval) training tasks.

Task Formats trained on:
- Classification
- Regression
- Proximity (Retrieval)
- Adhoc Search

  
It builds on the work done in [SciRepEval: A Multi-Format Benchmark for Scientific Document Representations](https://api.semanticscholar.org/CorpusID:254018137) and we evaluate the trained model on this benchmark as well.



- **Developed by:** Amanpreet Singh, Mike D'Arcy, Arman Cohan, Doug Downey, Sergey Feldman
- **Shared by :** Allen AI
- **Model type:** bert-base-uncased + adapters
- **License:** Apache 2.0
- **Finetuned from model:** [allenai/scibert](https://huggingface.co/allenai/scibert_scivocab_uncased).

## Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [https://github.com/allenai/SPECTER2](https://github.com/allenai/SPECTER2)
- **Paper:** [https://api.semanticscholar.org/CorpusID:254018137](https://api.semanticscholar.org/CorpusID:254018137)
- **Demo:** [Usage](https://github.com/allenai/SPECTER2/blob/main/README.md)

# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

## Direct Use

|Model|Name and HF link|Description|
|--|--|--|
|Proximity*|[allenai/specter2](https://huggingface.co/allenai/specter2)|Encode papers as queries and candidates eg. Link Prediction, Nearest Neighbor Search|
|Adhoc Query|[allenai/specter2_adhoc_query](https://huggingface.co/allenai/specter2_adhoc_query)|Encode short raw text queries for search tasks. (Candidate papers can be encoded with the proximity adapter)|
|Classification|[allenai/specter2_classification](https://huggingface.co/allenai/specter2_classification)|Encode papers to feed into linear classifiers as features|
|Regression|[allenai/specter2_regression](https://huggingface.co/allenai/specter2_regression)|Encode papers to feed into linear regressors as features|

*Proximity model should suffice for downstream task types not mentioned above

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

#load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
#other possibilities: allenai/specter2_<classification|regression|adhoc_query>

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# preprocess the input
inputs = self.tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
output = model(**inputs)
# take the first token in the batch as the embedding
embeddings = output.last_hidden_state[:, 0, :]
```

## Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

For evaluation and downstream usage, please refer to [https://github.com/allenai/scirepeval/blob/main/evaluation/INFERENCE.md](https://github.com/allenai/scirepeval/blob/main/evaluation/INFERENCE.md).

# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The base model is trained on citation links between papers and the adapters are trained on 8 large scale tasks across the four formats.
All the data is a part of SciRepEval benchmark and is available [here](https://huggingface.co/datasets/allenai/scirepeval).

The citation link are triplets in the form 

```json
{"query": {"title": ..., "abstract": ...}, "pos": {"title": ..., "abstract": ...}, "neg": {"title": ..., "abstract": ...}}
```

consisting of a query paper, a positive citation and a negative which can be from the same/different field of study as the query or citation of a citation.

## Training Procedure 

Please refer to the [SPECTER paper](https://api.semanticscholar.org/CorpusID:215768677).


### Training Hyperparameters


The model is trained in two stages using [SciRepEval](https://github.com/allenai/scirepeval/blob/main/training/TRAINING.md):
- Base Model: First a base model is trained on the above citation triplets.
``` batch size = 1024, max input length = 512, learning rate = 2e-5, epochs = 2 warmup steps = 10% fp16```
- Adapters: Thereafter, task format specific adapters are trained on the SciRepEval training tasks, where 600K triplets are sampled from above and added to the training data as well.
``` batch size = 256, max input length = 512, learning rate = 1e-4, epochs = 6 warmup = 1000 steps fp16```


# Evaluation

We evaluate the model on [SciRepEval](https://github.com/allenai/scirepeval), a large scale eval benchmark for scientific embedding tasks which which has [SciDocs] as a subset.
We also evaluate and establish a new SoTA on [MDCR](https://github.com/zoranmedic/mdcr), a large scale citation recommendation benchmark.

|Model|SciRepEval In-Train|SciRepEval Out-of-Train|SciRepEval Avg|MDCR(MAP, Recall@5)|
|--|--|--|--|--|
|[BM-25](https://api.semanticscholar.org/CorpusID:252199740)|n/a|n/a|n/a|(33.7, 28.5)|
|[SPECTER](https://huggingface.co/allenai/specter)|54.7|57.4|68.0|(30.6, 25.5)|
|[SciNCL](https://huggingface.co/malteos/scincl)|55.6|57.8|69.0|(32.6, 27.3)|
|[SciRepEval-Adapters](https://huggingface.co/models?search=scirepeval)|61.9|59.0|70.9|(35.3, 29.6)|
|[SPECTER2 Base](allenai/specter2_base)|56.3|73.6|69.1|(38.0, 32.4)|
|[SPECTER2-Adapters](https://huggingface.co/models?search=allenai/specter-2)|**62.3**|**59.2**|**71.2**|**(38.4, 33.0)**|

Please cite the following works if you end up using SPECTER2:

```
[SciRepEval paper](https://api.semanticscholar.org/CorpusID:254018137)
```bibtex
@article{Singh2022SciRepEvalAM,
  title={SciRepEval: A Multi-Format Benchmark for Scientific Document Representations},
  author={Amanpreet Singh and Mike D'Arcy and Arman Cohan and Doug Downey and Sergey Feldman},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.13308}
}
```


