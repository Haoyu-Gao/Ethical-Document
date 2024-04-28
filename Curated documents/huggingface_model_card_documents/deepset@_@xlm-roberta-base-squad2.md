---
license: cc-by-4.0
datasets:
- squad_v2
model-index:
- name: deepset/xlm-roberta-base-squad2
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
      value: 74.0354
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWMxNWQ2ODJkNWIzZGQwOWI4OTZjYjU3ZDVjZGQzMjI5MzljNjliZTY4Mzk4YTk4OTMzZWYxZjUxYmZhYTBhZSIsInZlcnNpb24iOjF9.eEeFYYJ30BfJDd-JYfI1kjlxJrRF6OFtj2GnkTCOO4kqX31inFy8ptDWusVlLFsUphm4dNWfTKXC5e-gytLBDA
    - type: f1
      value: 77.1833
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjg4MjNkOTA4Y2I5OGFlYTk1NWZjMWFlNjI5M2Y0NGZhMThhN2M4YmY2Y2RhZjcwYzU0MGNjN2RkZDljZmJmNiIsInZlcnNpb24iOjF9.TX42YMXpH4e0qu7cC4ARDlZWSkd55dwwyeyFXmOlXERNnEicDuFBCsy8WHLaqQCLUkzODJ22Hw4zhv81rwnlAQ
---

# Multilingual XLM-RoBERTa base for QA on various languages 

## Overview
**Language model:** xlm-roberta-base  
**Language:** Multilingual  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0   
**Eval data:** SQuAD 2.0 dev set - German MLQA - German XQuAD   
**Code:**  See [example](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering.py) in [FARM](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering.py)  
**Infrastructure**: 4x Tesla v100

## Hyperparameters

```
batch_size = 22*4
n_epochs = 2
max_seq_len=256,
doc_stride=128,
learning_rate=2e-5,
``` 

Corresponding experiment logs in mlflow: [link](https://public-mlflow.deepset.ai/#/experiments/2/runs/b25ec75e07614accb3f1ce03d43dbe08)


## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
```
"exact": 73.91560683904657
"f1": 77.14103746689592
```

Evaluated on German MLQA: test-context-de-question-de.json
  "exact": 33.67279167589108
  "f1": 44.34437105434842
  "total": 4517

Evaluated on German XQuAD: xquad.de.json
"exact": 48.739495798319325
  "f1": 62.552615701071495
  "total": 1190


## Usage

### In Transformers
```python
from transformers.pipelines import pipeline
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer

model_name = "deepset/xlm-roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
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

model_name = "deepset/xlm-roberta-base-squad2"

# a) Get predictions
nlp = Inferencer.load(model_name, task_type="question_answering")
QA_input = [{"questions": ["Why is model conversion important?"],
             "text": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."}]
res = nlp.inference_from_dicts(dicts=QA_input, rest_api_schema=True)

# b) Load model & tokenizer
model = AdaptiveModel.convert_from_transformers(model_name, device="cpu", task_type="question_answering")
tokenizer = Tokenizer.load(model_name)
```

### In haystack
For doing QA at scale (i.e. many docs instead of single paragraph), you can load the model also in [haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/xlm-roberta-base-squad2")
# or 
reader = TransformersReader(model="deepset/roberta-base-squad2",tokenizer="deepset/xlm-roberta-base-squad2")
```


## Authors
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
