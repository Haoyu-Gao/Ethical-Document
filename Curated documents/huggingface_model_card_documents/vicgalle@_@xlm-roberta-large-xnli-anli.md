---
language: multilingual
license: mit
tags:
- zero-shot-classification
- nli
- pytorch
datasets:
- mnli
- xnli
- anli
pipeline_tag: zero-shot-classification
widget:
- text: De pugna erat fantastic. Nam Crixo decem quam dilexit et praeciderunt caput
    aemulus.
  candidate_labels: violent, peaceful
- text: La película empezaba bien pero terminó siendo un desastre.
  candidate_labels: positivo, negativo, neutral
- text: La película empezó siendo un desastre pero en general fue bien.
  candidate_labels: positivo, negativo, neutral
- text: ¿A quién vas a votar en 2020?
  candidate_labels: Europa, elecciones, política, ciencia, deportes
---

### XLM-RoBERTa-large-XNLI-ANLI

XLM-RoBERTa-large model finetunned over several NLI datasets, ready to use for zero-shot classification.

Here are the accuracies for several test datasets:

|                             | XNLI-es | XNLI-fr | ANLI-R1 | ANLI-R2 | ANLI-R3 |
|-----------------------------|---------|---------|---------|---------|---------|
| xlm-roberta-large-xnli-anli | 93.7% | 93.2% | 68.5%  | 53.6%  | 49.0%  |

The model can be loaded with the zero-shot-classification pipeline like so:
```
from transformers import pipeline
classifier = pipeline("zero-shot-classification", 
                       model="vicgalle/xlm-roberta-large-xnli-anli")
```
You can then use this pipeline to classify sequences into any of the class names you specify:
```
sequence_to_classify = "Algún día iré a ver el mundo"
candidate_labels = ['viaje', 'cocina', 'danza']
classifier(sequence_to_classify, candidate_labels)
#{'sequence': 'Algún día iré a ver el mundo',
#'labels': ['viaje', 'danza', 'cocina'],
#'scores': [0.9991760849952698, 0.0004178212257102132, 0.0004059972707182169]}
```