---
language:
- de
- en
- multilingual
license: mit
tags:
- english
- german
---

# Bilingual English + German SQuAD2.0

We created German Squad 2.0 (**deQuAD 2.0**) and merged with [**SQuAD2.0**](https://rajpurkar.github.io/SQuAD-explorer/) into an English and German training data for question answering. The [**bert-base-multilingual-cased**](https://github.com/google-research/bert/blob/master/multilingual.md) is used to fine-tune bilingual QA downstream task.

## Details of deQuAD 2.0
[**SQuAD2.0**](https://rajpurkar.github.io/SQuAD-explorer/) was auto-translated into German. We hired professional editors to proofread the translated transcripts, correct mistakes and double check the answers to further polish the text and enhance annotation quality. The final German deQuAD dataset contains **130k** training and **11k** test samples.

## Overview
- **Language model:** bert-base-multilingual-cased  
- **Language:** German, English  
- **Training data:** deQuAD2.0 + SQuAD2.0 training set  
- **Evaluation data:** SQuAD2.0 test set; deQuAD2.0 test set
- **Infrastructure:** 8xV100 GPU  
- **Published**: July 9th, 2021

## Evaluation on English SQuAD2.0 

```
HasAns_exact = 85.79622132253711
HasAns_f1 = 90.92004586077663
HasAns_total = 5928
NoAns_exact = 94.76871320437343
NoAns_f1 = 94.76871320437343
NoAns_total = 5945
exact = 90.28889076054915
f1 = 92.84713483219753
total = 11873
```
## Evaluation on German deQuAD2.0 

```
HasAns_exact = 63.80526406330638
HasAns_f1 = 72.47269140789888
HasAns_total = 5813
NoAns_exact = 82.0291893792861
NoAns_f1 = 82.0291893792861
NoAns_total = 5687
exact = 72.81739130434782
f1 = 77.19858740470603
total = 11500
```
## Use Model in Pipeline


```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deutsche-telekom/bert-multi-english-german-squad2",
    tokenizer="deutsche-telekom/bert-multi-english-german-squad2"
)

contexts = ["Die Allianz Arena ist ein Fußballstadion im Norden von München und bietet bei Bundesligaspielen 75.021 Plätze, zusammengesetzt aus 57.343 Sitzplätzen, 13.794 Stehplätzen, 1.374 Logenplätzen, 2.152 Business Seats und 966 Sponsorenplätzen. In der Allianz Arena bestreitet der FC Bayern München seit der Saison 2005/06 seine Heimspiele. Bis zum Saisonende 2017 war die Allianz Arena auch Spielstätte des TSV 1860 München.",
            "Harvard is a large, highly residential research university. It operates several arts, cultural, and scientific museums, alongside the Harvard Library, which is the world's largest academic and private library system, comprising 79 individual libraries with over 18 million volumes. "]
questions = ["Wo befindet sich die Allianz Arena?", 
            "What is the worlds largest academic and private library system?"]
 
qa_pipeline(context=contexts, question=questions)

```

# Output:

```json
[{'score': 0.7290093898773193,
  'start': 44,
  'end': 62,
  'answer': 'Norden von München'},
 {'score': 0.7979822754859924,
  'start': 134,
  'end': 149,
  'answer': 'Harvard Library'}]
```
## License - The MIT License
Copyright (c) 2021 Fang Xu, Deutsche Telekom AG 
