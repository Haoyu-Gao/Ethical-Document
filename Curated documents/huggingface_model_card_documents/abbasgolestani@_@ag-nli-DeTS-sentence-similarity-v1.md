---
language:
- en
- nl
- de
- fr
- it
- es
license: apache-2.0
tags:
- feature-extraction
- sentence-similarity
- transformers
datasets:
- multi_nli
- pietrolesci/nli_fever
pipeline_tag: text-classification
---

# Cross-Encoder for Sentence Similarity
This model was trained using [SentenceTransformers](https://sbert.net) [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) class.

## Training Data
This model was trained on 6 different nli datasets. The model will predict a score between 0 (not similar) and 1 (very similar) for the semantic similarity of two sentences. 


## Usage (CrossEncoder)
Comparing each sentence of sentences1 array to the corrosponding sentence of sentences2 array like comparing the first sentnece of each array, then comparing the second sentence of each array,...
```python
from sentence_transformers import CrossEncoder


model = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')

# Two lists of sentences
sentences1 = ['I am honored to be given the opportunity to help make our company better',
             'I love my job and what I do here',
             'I am excited about our company’s vision']

sentences2 = ['I am hopeful about the future of our company',
              'My work is aligning with my passion',
              'Definitely our company vision will be the next breakthrough to change the world and I’m so happy and proud to work here']

pairs = zip(sentences1,sentences2)
list_pairs=list(pairs)

scores1 = model.predict(list_pairs, show_progress_bar=False)
print(scores1)

for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], scores1[i]))

```





## Usage #2

Pre-trained models can be used like this:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')
scores = model.predict([('Sentence 1', 'Sentence 2'), ('Sentence 3', 'Sentence 4')])
```

The model will predict scores for the pairs `('Sentence 1', 'Sentence 2')` and `('Sentence 3', 'Sentence 4')`.

You can use this model also without sentence_transformers and by just using Transformers ``AutoModel`` class