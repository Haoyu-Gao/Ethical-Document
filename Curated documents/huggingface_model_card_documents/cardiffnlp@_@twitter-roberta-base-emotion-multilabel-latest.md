---
language:
- en
tags:
- generated_from_keras_callback
pipeline_tag: text-classification
model-index:
- name: twitter-roberta-base-emotion-multilabel-latest
  results: []
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# twitter-roberta-base-emotion-multilabel-latest

This model is a fine-tuned version of [cardiffnlp/twitter-roberta-base-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2022-154m) on the 
[`SemEval 2018 - Task 1 Affect in Tweets`](https://aclanthology.org/S18-1001/) `(subtask: E-c / multilabel classification)`.



## Performance

Following metrics are achieved on the test split:

- F1 (micro): 0.7169
- F1 (macro): 0.5464  
- Jaccard Index (samples): 0.5970: 

### Usage
#### 1. [tweetnlp](https://pypi.org/project/tweetnlp/)
Install tweetnlp via pip.
```shell
pip install tweetnlp
```
Load the model in python.
```python
import tweetnlp

model = tweetnlp.load_model('topic_classification', model_name='cardiffnlp/twitter-roberta-base-emotion-multilabel-latest')

model.predict("I bet everything will work out in the end :)")

>> {'label': ['joy', 'optimism']}

```
#### 2. pipeline
```shell
pip install -U tensorflow==2.10
```

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", return_all_scores=True)

pipe("I bet everything will work out in the end :)")

>> [[{'label': 'anger', 'score': 0.018903767690062523},
  {'label': 'anticipation', 'score': 0.28172484040260315},
  {'label': 'disgust', 'score': 0.011607927270233631},
  {'label': 'fear', 'score': 0.036411102861166},
  {'label': 'joy', 'score': 0.8812029361724854},
  {'label': 'love', 'score': 0.09591569006443024},
  {'label': 'optimism', 'score': 0.9810988306999207},
  {'label': 'pessimism', 'score': 0.016823478043079376},
  {'label': 'sadness', 'score': 0.01889917254447937},
  {'label': 'surprise', 'score': 0.02702752873301506},
  {'label': 'trust', 'score': 0.4155798852443695}]]
```


### Reference 
```
@inproceedings{camacho-collados-etal-2022-tweetnlp,
    title={{T}weet{NLP}: {C}utting-{E}dge {N}atural {L}anguage {P}rocessing for {S}ocial {M}edia},
    author={Camacho-Collados, Jose and Rezaee, Kiamehr and Riahi, Talayeh and Ushio, Asahi and Loureiro, Daniel and Antypas, Dimosthenis and Boisson, Joanne and Espinosa-Anke, Luis and Liu, Fangyu and Mart{\'\i}nez-C{\'a}mara, Eugenio and others},
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2022",
    address = "Abu Dhabi, U.A.E.",
    publisher = "Association for Computational Linguistics",
}

```