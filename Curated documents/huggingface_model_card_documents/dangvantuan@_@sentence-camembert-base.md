---
language: fr
license: apache-2.0
tags:
- Text
- Sentence Similarity
- Sentence-Embedding
- camembert-base
datasets:
- stsb_multi_mt
pipeline_tag: sentence-similarity
model-index:
- name: sentence-camembert-base by Van Tuan DANG
  results:
  - task:
      type: Text Similarity
      name: Sentence-Embedding
    dataset:
      name: Text Similarity fr
      type: stsb_multi_mt
      args: fr
    metrics:
    - type: Pearson_correlation_coefficient
      value: xx.xx
      name: Test Pearson correlation coefficient
---

## Pre-trained sentence embedding models are the state-of-the-art of Sentence Embeddings for French.
Model is Fine-tuned using pre-trained [facebook/camembert-base](https://huggingface.co/camembert/camembert-base) and
[Siamese BERT-Networks with 'sentences-transformers'](https://www.sbert.net/) on dataset [stsb](https://huggingface.co/datasets/stsb_multi_mt/viewer/fr/train)


## Usage
The model can be used directly (without a language model) as follows:

```python
from sentence_transformers import SentenceTransformer
model =  SentenceTransformer("dangvantuan/sentence-camembert-base")

sentences = ["Un avion est en train de décoller.",
          "Un homme joue d'une grande flûte.",
          "Un homme étale du fromage râpé sur une pizza.",
          "Une personne jette un chat au plafond.",
          "Une personne est en train de plier un morceau de papier.",
          ]

embeddings = model.encode(sentences)
```

## Evaluation
The model can be evaluated as follows on the French test data of stsb.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset
def convert_dataset(dataset):
    dataset_samples=[]
    for df in dataset:
        score = float(df['similarity_score'])/5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[df['sentence1'], 
                                    df['sentence2']], label=score)
        dataset_samples.append(inp_example)
    return dataset_samples

# Loading the dataset for evaluation
df_dev = load_dataset("stsb_multi_mt", name="fr", split="dev")
df_test = load_dataset("stsb_multi_mt", name="fr", split="test")

# Convert the dataset for evaluation

# For Dev set:
dev_samples = convert_dataset(df_dev)
val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
val_evaluator(model, output_path="./")

# For Test set:
test_samples = convert_dataset(df_test)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path="./")
```

**Test Result**: 
The performance is measured using Pearson and Spearman correlation:
- On dev


| Model  | Pearson correlation | Spearman correlation  |  #params  |
| ------------- | ------------- | ------------- |------------- |
| [dangvantuan/sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base)| 86.73 |86.54 | 110M |
| [distiluse-base-multilingual-cased](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased) | 79.22 | 79.16|135M |
- On test


| Model  | Pearson correlation | Spearman correlation  |
| ------------- | ------------- | ------------- |
| [dangvantuan/sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base)| 82.36 | 81.64|
| [distiluse-base-multilingual-cased](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased) | 78.62 | 77.48|


## Citation


	@article{reimers2019sentence,
	   title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
	   author={Nils Reimers, Iryna Gurevych},
	   journal={https://arxiv.org/abs/1908.10084},
	   year={2019}
	}


	@article{martin2020camembert,
	   title={CamemBERT: a Tasty French Language Mode},
	   author={Martin, Louis and Muller, Benjamin and Su{\'a}rez, Pedro Javier Ortiz and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^\i}t},
	   journal={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
	   year={2020}
	}