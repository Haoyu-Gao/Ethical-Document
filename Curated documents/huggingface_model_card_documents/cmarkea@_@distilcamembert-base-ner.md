---
language: fr
license: mit
datasets:
- Jean-Baptiste/wikiner_fr
widget:
- text: Boulanger, habitant à Boulanger et travaillant dans le magasin Boulanger situé
    dans la ville de Boulanger. Boulanger a écrit le livre éponyme Boulanger édité
    par la maison d'édition Boulanger.
- text: Quentin Jerome Tarantino naît le 27 mars 1963 à Knoxville, dans le Tennessee.
    Il est le fils de Connie McHugh, une infirmière, née le 3 septembre 1946, et de
    Tony Tarantino, acteur et musicien amateur né à New York. Ce dernier est d'origine
    italienne par son père ; sa mère a des ascendances irlandaises et cherokees. Il
    est prénommé d'après Quint Asper, le personnage joué par Burt Reynolds dans la
    série Gunsmoke et Quentin Compson, personnage du roman Le Bruit et la Fureur.
    Son père quitte le domicile familial avant même sa naissance. En 1965, sa mère
    déménage à Torrance, dans la banlieue sud de Los Angeles, et se remarie avec Curtis
    Zastoupil, un pianiste de bar, qui lui fait découvrir le cinéma. Le couple divorce
    alors que le jeune Quentin a une dizaine d'années.
---
DistilCamemBERT-NER
===================

We present DistilCamemBERT-NER, which is [DistilCamemBERT](https://huggingface.co/cmarkea/distilcamembert-base) fine-tuned for the NER (Named Entity Recognition) task for the French language. The work is inspired by [Jean-Baptiste/camembert-ner](https://huggingface.co/Jean-Baptiste/camembert-ner) based on the [CamemBERT](https://huggingface.co/camembert-base) model. The problem of the modelizations based on CamemBERT is at the scaling moment, for the production phase, for example. Indeed, inference cost can be a technological issue. To counteract this effect, we propose this modelization which **divides the inference time by two** with the same consumption power thanks to [DistilCamemBERT](https://huggingface.co/cmarkea/distilcamembert-base).

Dataset
-------

The dataset used is [wikiner_fr](https://huggingface.co/datasets/Jean-Baptiste/wikiner_fr), which represents ~170k sentences labeled in 5 categories :
* PER: personality ;
* LOC: location ;
* ORG: organization ;
* MISC: miscellaneous entities (movies title, books, etc.) ;
* O: background (Outside entity).
 
Evaluation results
------------------

| **class**      | **precision (%)** | **recall (%)** | **f1 (%)** | **support (#sub-word)** |
| :------------: | :---------------: | :------------: | :--------: | :---------------------: |
| **global**     | 98.17             | 98.19          | 98.18      | 378,776                 |
| **PER**        | 96.78             | 96.87          | 96.82      | 23,754                  |
| **LOC**        | 94.05             | 93.59          | 93.82      | 27,196                  |
| **ORG**        | 86.05             | 85.92          | 85.98      | 6,526                   |
| **MISC**       | 88.78             | 84.69          | 86.69      | 11,891                  |
| **O**          | 99.26             | 99.47          | 99.37      | 309,409                 |

Benchmark
---------

This model performance is compared to 2 reference models (see below) with the metric f1 score. For the mean inference time measure, an AMD Ryzen 5 4500U @ 2.3GHz with 6 cores was used:

| **model**                                                                                                         | **time (ms)** | **PER (%)** | **LOC (%)** | **ORG (%)** | **MISC (%)**  | **O (%)** |
| :---------------------------------------------------------------------------------------------------------------: | :-----------: | :---------: | :---------: | :---------: | :-----------: | :-------: |
| [cmarkea/distilcamembert-base-ner](https://huggingface.co/cmarkea/distilcamembert-base-ner)                       | **43.44** | **96.82** | **93.82** | **85.98** | **86.69** | **99.37** |
| [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) | 87.56  | 79.93 | 72.89 | 61.34 | n/a   | 96.04 |
| [flair/ner-french](https://huggingface.co/flair/ner-french)                                                       | 314.96 | 82.91 | 76.17 | 70.96 | 76.29 | 97.65 |

How to use DistilCamemBERT-NER
------------------------------

```python
from transformers import pipeline

ner = pipeline(
    task='ner',
    model="cmarkea/distilcamembert-base-ner",
    tokenizer="cmarkea/distilcamembert-base-ner",
    aggregation_strategy="simple"
)
result = ner(
    "Le Crédit Mutuel Arkéa est une banque Française, elle comprend le CMB "
    "qui est une banque située en Bretagne et le CMSO qui est une banque "
    "qui se situe principalement en Aquitaine. C'est sous la présidence de "
    "Louis Lichou, dans les années 1980 que différentes filiales sont créées "
    "au sein du CMB et forment les principales filiales du groupe qui "
    "existent encore aujourd'hui (Federal Finance, Suravenir, Financo, etc.)."
)

result
[{'entity_group': 'ORG',
  'score': 0.9974479,
  'word': 'Crédit Mutuel Arkéa',
  'start': 3,
  'end': 22},
 {'entity_group': 'LOC',
  'score': 0.9000358,
  'word': 'Française',
  'start': 38,
  'end': 47},
 {'entity_group': 'ORG',
  'score': 0.9788757,
  'word': 'CMB',
  'start': 66,
  'end': 69},
 {'entity_group': 'LOC',
  'score': 0.99919766,
  'word': 'Bretagne',
  'start': 99,
  'end': 107},
 {'entity_group': 'ORG',
  'score': 0.9594884,
  'word': 'CMSO',
  'start': 114,
  'end': 118},
 {'entity_group': 'LOC',
  'score': 0.99935514,
  'word': 'Aquitaine',
  'start': 169,
  'end': 178},
 {'entity_group': 'PER',
  'score': 0.99911094,
  'word': 'Louis Lichou',
  'start': 208,
  'end': 220},
 {'entity_group': 'ORG',
  'score': 0.96226394,
  'word': 'CMB',
  'start': 291,
  'end': 294},
 {'entity_group': 'ORG',
  'score': 0.9983959,
  'word': 'Federal Finance',
  'start': 374,
  'end': 389},
 {'entity_group': 'ORG',
  'score': 0.9984454,
  'word': 'Suravenir',
  'start': 391,
  'end': 400},
 {'entity_group': 'ORG',
  'score': 0.9985084,
  'word': 'Financo',
  'start': 402,
  'end': 409}]
```

### Optimum + ONNX
```python
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer, pipeline

HUB_MODEL = "cmarkea/distilcamembert-base-nli"
tokenizer = AutoTokenizer.from_pretrained(HUB_MODEL)
model = ORTModelForTokenClassification.from_pretrained(HUB_MODEL)
onnx_qa = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Quantized onnx model
quantized_model = ORTModelForTokenClassification.from_pretrained(
    HUB_MODEL, file_name="model_quantized.onnx"
)
```

Citation
--------
```bibtex
@inproceedings{delestre:hal-03674695,
  TITLE = {{DistilCamemBERT : une distillation du mod{\`e}le fran{\c c}ais CamemBERT}},
  AUTHOR = {Delestre, Cyrile and Amar, Abibatou},
  URL = {https://hal.archives-ouvertes.fr/hal-03674695},
  BOOKTITLE = {{CAp (Conf{\'e}rence sur l'Apprentissage automatique)}},
  ADDRESS = {Vannes, France},
  YEAR = {2022},
  MONTH = Jul,
  KEYWORDS = {NLP ; Transformers ; CamemBERT ; Distillation},
  PDF = {https://hal.archives-ouvertes.fr/hal-03674695/file/cap2022.pdf},
  HAL_ID = {hal-03674695},
  HAL_VERSION = {v1},
}
```