---
language: de
license: mit
tags:
- german
---

We released the German Question Answering model fine-tuned with our own German Question Answering dataset (**deQuAD**) containing **130k** training and **11k** test QA pairs.

## Overview
- **Language model:** [electra-base-german-uncased](https://huggingface.co/german-nlp-group/electra-base-german-uncased)
- **Language:** German
- **Training data:** deQuAD2.0 training set (~42MB)
- **Evaluation data:** deQuAD2.0 test set (~4MB)
- **Infrastructure:** 8xV100 GPU  

## Evaluation
We benchmarked the question answering performance on our deQuAD test data with some German language models. The fine-tuned electra-base-german-uncased model gives the best performance (Exact Match/F1).


| Model | All | HasAns | NoAns |  
|-------|--------|--------|--------|
| electra-base-german-uncased | 70.97/76.18 | 67.73/78.02 | 74.29/74.29 |
| bert-base-german-cased |58.98/64.77| 49.19/60.63| 69.03/69.03|
|bert-base-german-dbmdz-uncased|63.70/68.00| 57.03/65.52| 70.51/70.51 |
|dbmdz/bert-base-german-europeana-uncased| 58.79/63.38| 52.14/61.22| 65.59/65.59|

## Use Model in Pipeline

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deutsche-telekom/electra-base-de-squad2",
    tokenizer="deutsche-telekom/electra-base-de-squad2"
)

contexts = ['''Die Robert Bosch GmbH ist ein im Jahr 1886 von Robert Bosch gegründetes multinationales deutsches Unternehmen. 
Es ist tätig als Automobilzulieferer, Hersteller von Gebrauchsgütern und Industrie- und Gebäudetechnik und darüber hinaus 
in der automatisierten Verpackungstechnik, wo Bosch den führenden Platz einnimmt. Die Robert Bosch GmbH und ihre rund 460 
Tochter- und Regionalgesellschaften in mehr als 60 Ländern bilden die Bosch-Gruppe. Der Sitz der Geschäftsführung befindet 
sich auf der Schillerhöhe in Gerlingen, der Firmensitz in Stuttgart. Seit dem 1. Juli 2012 ist Volkmar Denner Vorsitzender 
der Geschäftsführung. Im Jahr 2015 konnte Bosch die Spitzenposition zurückgewinnen. Die Automobilsparte war im Jahr 2018 
für 61 % des Konzernumsatzes von Bosch verantwortlich. Das Unternehmen hatte im Jahr 2018 in Deutschland an 85 Standorten 
139.400 Mitarbeiter.''']*2

questions = ["Wer leitet die Robert Bosch GmbH?", 
            "Wer begründete die Robert Bosch GmbH?"]

qa_pipeline(context=contexts, question=questions)
```

## Output
```json
[{'score': 0.9537325501441956,
  'start': 577,
  'end': 591,
  'answer': 'Volkmar Denner'},
 {'score': 0.8804352879524231,
  'start': 47,
  'end': 59,
  'answer': 'Robert Bosch'}]
```

## License - The MIT License
Copyright (c) 2021 Fang Xu, Deutsche Telekom AG 
