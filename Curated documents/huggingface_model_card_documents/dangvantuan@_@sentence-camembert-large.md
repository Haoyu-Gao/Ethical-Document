---
language: fr
license: apache-2.0
tags:
- Text
- Sentence Similarity
- Sentence-Embedding
- camembert-large
datasets:
- stsb_multi_mt
pipeline_tag: sentence-similarity
model-index:
- name: sentence-camembert-large by Van Tuan DANG
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
## Description:
[**Sentence-CamemBERT-Large**](https://huggingface.co/dangvantuan/sentence-camembert-large) is the Embedding Model for French developed by [La Javaness](https://www.lajavaness.com/). The purpose of this embedding model is to represent the content and semantics of a French sentence in a mathematical vector which allows it to understand the meaning of the text-beyond individual words in queries and documents, offering a powerful semantic search.
## Pre-trained sentence embedding models are state-of-the-art of Sentence Embeddings for French.
The model is Fine-tuned using pre-trained [facebook/camembert-large](https://huggingface.co/camembert/camembert-large) and
[Siamese BERT-Networks with 'sentences-transformers'](https://www.sbert.net/) on dataset [stsb](https://huggingface.co/datasets/stsb_multi_mt/viewer/fr/train)


## Usage
The model can be used directly (without a language model) as follows:

```python
from sentence_transformers import SentenceTransformer
model =  SentenceTransformer("dangvantuan/sentence-camembert-large")

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


| Model  | Pearson correlation | Spearman correlation  | #params  |
| ------------- | ------------- | ------------- |------------- |
| [dangvantuan/sentence-camembert-large](https://huggingface.co/dangvantuan/sentence-camembert-large)| 88.2 |88.02 | 336M| 
| [dangvantuan/sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base)  | 86.73|86.54 | 110M |
| [distiluse-base-multilingual-cased](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased) | 79.22 | 79.16|135M |
| [GPT-3 (text-davinci-003)](https://platform.openai.com/docs/models) | 85 | NaN|175B |
| [GPT-(text-embedding-ada-002)](https://platform.openai.com/docs/models) | 79.75 | 80.44|NaN |
- On test


| Model  | Pearson correlation | Spearman correlation  | 
| ------------- | ------------- | ------------- |
| [dangvantuan/sentence-camembert-large](https://huggingface.co/dangvantuan/sentence-camembert-large)| 85.9 | 85.8|
| [dangvantuan/sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base)| 82.36 | 81.64|
| [distiluse-base-multilingual-cased](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased) | 78.62 | 77.48|
| [GPT-3 (text-davinci-003)](https://platform.openai.com/docs/models) | 82 | NaN|175B |
| [GPT-(text-embedding-ada-002)](https://platform.openai.com/docs/models) | 79.05 | 77.56|NaN |


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