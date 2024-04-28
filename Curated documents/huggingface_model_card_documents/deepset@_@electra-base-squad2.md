---
language: en
license: cc-by-4.0
datasets:
- squad_v2
model-index:
- name: deepset/electra-base-squad2
  results:
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squad_v2
      type: squad_v2
      config: squad_v2
      split: validation
    metrics:
    - type: exact_match
      value: 77.6074
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYzE5NTRmMmUwYTk1MTI0NjM0ZmQwNDFmM2Y4Mjk4ZWYxOGVmOWI3ZGFiNWM4OTUxZDQ2ZjdmNmU3OTk5ZjRjYyIsInZlcnNpb24iOjF9.0VZRewdiovE4z3K5box5R0oTT7etpmd0BX44FJBLRFfot-uJ915b-bceSv3luJQ7ENPjaYSa7o7jcHlDzn3oAw
    - type: f1
      value: 81.7181
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2VlMzM0Y2UzYjhhNTJhMTFiYWZmMDNjNjRiZDgwYzc5NWE3N2M4ZGFlYWQ0ZjVkZTE2MDU0YmMzMDc1MTY5MCIsInZlcnNpb24iOjF9.jRV58UxOM7CJJSsmxJuZvlt00jMGA1thp4aqtcFi1C8qViQ1kW7NYz8rg1gNTDZNez2UwPS1NgN_HnnwBHPbCQ
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squad
      type: squad
      config: plain_text
      split: validation
    metrics:
    - type: exact_match
      value: 80.407
      name: Exact Match
    - type: f1
      value: 88.942
      name: F1
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: adversarial_qa
      type: adversarial_qa
      config: adversarialQA
      split: validation
    metrics:
    - type: exact_match
      value: 23.533
      name: Exact Match
    - type: f1
      value: 36.521
      name: F1
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squad_adversarial
      type: squad_adversarial
      config: AddOneSent
      split: validation
    metrics:
    - type: exact_match
      value: 73.867
      name: Exact Match
    - type: f1
      value: 81.381
      name: F1
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squadshifts amazon
      type: squadshifts
      config: amazon
      split: test
    metrics:
    - type: exact_match
      value: 64.512
      name: Exact Match
    - type: f1
      value: 80.166
      name: F1
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squadshifts new_wiki
      type: squadshifts
      config: new_wiki
      split: test
    metrics:
    - type: exact_match
      value: 76.568
      name: Exact Match
    - type: f1
      value: 87.706
      name: F1
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squadshifts nyt
      type: squadshifts
      config: nyt
      split: test
    metrics:
    - type: exact_match
      value: 77.884
      name: Exact Match
    - type: f1
      value: 87.858
      name: F1
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: squadshifts reddit
      type: squadshifts
      config: reddit
      split: test
    metrics:
    - type: exact_match
      value: 64.399
      name: Exact Match
    - type: f1
      value: 78.096
      name: F1
---

# electra-base for QA

## Overview
**Language model:** electra-base  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  See [example](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering.py) in [FARM](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering.py)  
**Infrastructure**: 1x Tesla v100

## Hyperparameters

```
seed=42
batch_size = 32
n_epochs = 5
base_LM_model = "google/electra-base-discriminator"
max_seq_len = 384
learning_rate = 1e-4
lr_schedule = LinearWarmup
warmup_proportion = 0.1
doc_stride=128
max_query_length=64
```

## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
```
"exact": 77.30144024256717,
 "f1": 81.35438272008543,
 "total": 11873,
 "HasAns_exact": 74.34210526315789,
 "HasAns_f1": 82.45961302894314,
 "HasAns_total": 5928,
 "NoAns_exact": 80.25231286795626,
 "NoAns_f1": 80.25231286795626,
 "NoAns_total": 5945
```

## Usage

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/electra-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and lets people easily switch between frameworks.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### In FARM

```python
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.infer import Inferencer

model_name = "deepset/electra-base-squad2"

# a) Get predictions
nlp = Inferencer.load(model_name, task_type="question_answering")
QA_input = [{"questions": ["Why is model conversion important?"],
             "text": "The option to convert models between FARM and transformers gives freedom to the user and lets people easily switch between frameworks."}]
res = nlp.inference_from_dicts(dicts=QA_input)

# b) Load model & tokenizer
model = AdaptiveModel.convert_from_transformers(model_name, device="cpu", task_type="question_answering")
tokenizer = Tokenizer.load(model_name)
```

### In haystack
For doing QA at scale (i.e. many docs instead of a single paragraph), you can load the model also in [haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/electra-base-squad2")
# or
reader = TransformersReader(model="deepset/electra-base-squad2",tokenizer="deepset/electra-base-squad2")
```


## Authors
Vaishali Pal `vaishali.pal [at] deepset.ai`  
Branden Chan: `branden.chan [at] deepset.ai`  
Timo Möller: `timo.moeller [at] deepset.ai`  
Malte Pietsch: `malte.pietsch [at] deepset.ai`  
Tanay Soni: `tanay.soni [at] deepset.ai`  

## About us
![deepset logo](https://workablehr.s3.amazonaws.com/uploads/account/logo/476306/logo)

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  

Some of our work: 
- [German BERT (aka "bert-base-german-cased")](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR datasets and models (aka "gelectra-base-germanquad", "gbert-base-germandpr")](https://deepset.ai/germanquad)
- [FARM](https://github.com/deepset-ai/FARM)
- [Haystack](https://github.com/deepset-ai/haystack/)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](http://www.deepset.ai/jobs)
