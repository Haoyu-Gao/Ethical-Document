---
language:
- ja
license: mit
tags:
- generated_from_trainer
- ner
- bert
metrics:
- f1
widget:
- text: 鈴木は4月の陽気の良い日に、鈴をつけて熊本県の阿蘇山に登った
- text: 中国では、中国共産党による一党統治が続く
base_model: xlm-roberta-base
model-index:
- name: xlm-roberta-ner-ja
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# xlm-roberta-ner-japanese

(Japanese caption : 日本語の固有表現抽出のモデル)

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) (pre-trained cross-lingual ```RobertaModel```) trained for named entity recognition (NER) token classification.

The model is fine-tuned on NER dataset provided by Stockmark Inc, in which data is collected from Japanese Wikipedia articles.<br>
See [here](https://github.com/stockmarkteam/ner-wikipedia-dataset) for the license of this dataset.

Each token is labeled by :

| Label id | Tag | Tag in Widget | Description |
|---|---|---|---|
| 0 | O | (None) | others or nothing |
| 1 | PER | PER | person |
| 2 | ORG | ORG | general corporation organization |
| 3 | ORG-P | P | political organization |
| 4 | ORG-O | O | other organization |
| 5 | LOC | LOC | location |
| 6 | INS | INS | institution, facility |
| 7 | PRD | PRD | product |
| 8 | EVT | EVT | event |

## Intended uses

```python
from transformers import pipeline

model_name = "tsmatz/xlm-roberta-ner-japanese"
classifier = pipeline("token-classification", model=model_name)
result = classifier("鈴木は4月の陽気の良い日に、鈴をつけて熊本県の阿蘇山に登った")
print(result)
```

## Training procedure

You can download the source code for fine-tuning from [here](https://github.com/tsmatz/huggingface-finetune-japanese/blob/master/01-named-entity.ipynb).

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 12
- eval_batch_size: 12
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1     |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| No log        | 1.0   | 446  | 0.1510          | 0.8457 |
| No log        | 2.0   | 892  | 0.0626          | 0.9261 |
| No log        | 3.0   | 1338 | 0.0366          | 0.9580 |
| No log        | 4.0   | 1784 | 0.0196          | 0.9792 |
| No log        | 5.0   | 2230 | 0.0173          | 0.9864 |


### Framework versions

- Transformers 4.23.1
- Pytorch 1.12.1+cu102
- Datasets 2.6.1
- Tokenizers 0.13.1
