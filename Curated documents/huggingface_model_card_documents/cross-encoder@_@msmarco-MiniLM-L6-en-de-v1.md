---
license: apache-2.0
---
# Cross-Encoder for MS MARCO - EN-DE

This is a cross-lingual Cross-Encoder model for EN-DE that can be used for passage re-ranking. It was trained on the [MS Marco Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) task.

The model can be used for Information Retrieval:  See [SBERT.net Retrieve & Re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html). 

The training code is available in this repository, see `train_script.py`.


## Usage with SentenceTransformers

When you have [SentenceTransformers](https://www.sbert.net/) installed, you can use the model like this:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name', max_length=512)
query = 'How many people live in Berlin?'
docs = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
pairs = [(query, doc) for doc in docs]
scores = model.predict(pairs)
```


## Usage with Transformers
With the transformers library, you can use the model like this:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('model_name')
tokenizer = AutoTokenizer.from_pretrained('model_name')

features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
```




## Performance
The performance was evaluated on three datasets:
- **TREC-DL19 EN-EN**: The original [TREC 2019 Deep Learning Track](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html): Given an English query and 1000 documents (retrieved by BM25 lexical search), rank documents with according to their relevance. We compute NDCG@10. BM25 achieves a score of 45.46, a perfect re-ranker can achieve a score of 95.47. 
- **TREC-DL19 DE-EN**: The English queries of TREC-DL19 have been translated by a German native speaker to German. We rank the German queries versus the English passages from the original TREC-DL19 setup. We compute NDCG@10.
- **GermanDPR DE-DE**: The [GermanDPR](https://www.deepset.ai/germanquad) dataset provides German queries and German passages from Wikipedia. We indexed the 2.8 Million paragraphs from German Wikipedia and retrieved for each query the top 100 most relevant passages using BM25 lexical search with Elasticsearch. We compute MRR@10. BM25 achieves a score of 35.85, a perfect re-ranker can achieve a score of 76.27.

We also check the performance of bi-encoders using the same evaluation: The retrieved documents from BM25 lexical search are re-ranked using query & passage embeddings with cosine-similarity. Bi-Encoders can also be used for end-to-end semantic search.


| Model-Name | TREC-DL19 EN-EN | TREC-DL19 DE-EN | GermanDPR DE-DE | Docs / Sec |
| ------------- |:-------------:| :-----: | :---: | :----: |
| BM25 | 45.46 | - | 35.85 | -|
| **Cross-Encoder Re-Rankers** | | | |
| [cross-encoder/msmarco-MiniLM-L6-en-de-v1](https://huggingface.co/cross-encoder/msmarco-MiniLM-L6-en-de-v1) | 72.43 | 65.53 | 46.77 | 1600 |
| [cross-encoder/msmarco-MiniLM-L12-en-de-v1](https://huggingface.co/cross-encoder/msmarco-MiniLM-L12-en-de-v1) | 72.94 | 66.07 | 49.91 | 900 |
| [svalabs/cross-electra-ms-marco-german-uncased](https://huggingface.co/svalabs/cross-electra-ms-marco-german-uncased) (DE only) | - | - | 53.67 | 260 |
| [deepset/gbert-base-germandpr-reranking](https://huggingface.co/deepset/gbert-base-germandpr-reranking) (DE only) | - | - | 53.59 | 260 |
| **Bi-Encoders (re-ranking)** | | | |
| [sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned](https://huggingface.co/sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned) | 63.38 | 58.28 | 37.88 | 940 |
| [sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch](https://huggingface.co/sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch) | 65.51 | 58.69 | 38.32 | 940 |
| [svalabs/bi-electra-ms-marco-german-uncased](https://huggingface.co/svalabs/bi-electra-ms-marco-german-uncased) (DE only) | - | - | 34.31 | 450 |
| [deepset/gbert-base-germandpr-question_encoder](https://huggingface.co/deepset/gbert-base-germandpr-question_encoder) (DE only) | - | - | 42.55 | 450 |



 Note: Docs / Sec gives the number of (query, document) pairs we can re-rank within a second on a V100 GPU.
