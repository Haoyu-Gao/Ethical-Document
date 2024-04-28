---
language: en
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
---

# sentence-transformers/sentence-t5-xl

This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional dense vector space. The model works well for sentence similarity tasks, but doesn't perform that well for semantic search tasks.

This model was converted from the Tensorflow model [st5-3b-1](https://tfhub.dev/google/sentence-t5/st5-3b/1) to PyTorch. When using this model, have a look at the publication: [Sentence-T5: Scalable sentence encoders from pre-trained text-to-text models](https://arxiv.org/abs/2108.08877). The tfhub model and this PyTorch model can produce slightly different embeddings, however, when run on the same benchmarks, they produce identical results.

The model uses only the encoder from a T5-3B model. The weights are stored in FP16.  


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
embeddings = model.encode(sentences)
print(embeddings)
```

The model requires sentence-transformers version 2.2.0 or newer.

## Evaluation Results

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/sentence-t5-xl)



## Citing & Authors

If you find this model helpful, please cite the respective publication:
[Sentence-T5: Scalable sentence encoders from pre-trained text-to-text models](https://arxiv.org/abs/2108.08877)
