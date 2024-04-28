---
license: apache-2.0
---
# Cross-Encoder for Quora Duplicate Questions Detection
This model was trained using [SentenceTransformers](https://sbert.net) [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) class.

## Training Data
This model was trained on the [STS benchmark dataset](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark). The model will predict a score between 0 and 1 how for the semantic similarity of two sentences. 


## Usage and Performance

Pre-trained models can be used like this:
```
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name')
scores = model.predict([('Sentence 1', 'Sentence 2'), ('Sentence 3', 'Sentence 4')])
```

The model will predict scores for the pairs `('Sentence 1', 'Sentence 2')` and `('Sentence 3', 'Sentence 4')`.

You can use this model also without sentence_transformers and by just using Transformers ``AutoModel`` class