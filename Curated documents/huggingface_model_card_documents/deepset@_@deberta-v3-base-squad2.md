---
language: en
license: cc-by-4.0
tags:
- deberta
- deberta-v3
datasets:
- squad_v2
base_model: microsoft/deberta-v3-base
model-index:
- name: deepset/deberta-v3-base-squad2
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
      value: 83.8248
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2IyZTEyYzNlOTAwZmFlNWRiZTdiNzQzMTUyM2FmZTQ3ZWQwNWZmMzc2ZDVhYWYyMzkxOTUyMGNlMWY0M2E5MiIsInZlcnNpb24iOjF9.y8KvfefMLI977BYun0X1rAq5qudmezW_UJe9mh6sYBoiWaBosDO5TRnEGR1BHzdxmv2EgPK_PSomtZvb043jBQ
    - type: f1
      value: 87.41
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiOWVhNjAwM2Q5N2Y3MGU4ZWY3N2Y0MmNjYWYwYmQzNTdiYWExODhkYmQ1YjIwM2I1ODEzNWIxZDI1ZWQ1YWRjNSIsInZlcnNpb24iOjF9.Jk0v1ZheLRFz6k9iNAgCMMZtPYj5eVwUCku4E76wRYc-jHPmiUuxvNiNkn6NW-jkBD8bJGMqDSjJyVpVMn9pBA
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
      value: 84.9678
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiOWUxYTg4MzU3YTdmMDRmMGM0NjFjMTcwNGM3YzljM2RkMTc1ZGNhMDQwMTgwNGI0ZDE4ZGMxZTE3YjY5YzQ0ZiIsInZlcnNpb24iOjF9.KKaJ1UtikNe2g6T8XhLoWNtL9X4dHHyl_O4VZ5LreBT9nXneGc21lI1AW3n8KXTFGemzRpRMvmCDyKVDHucdDQ
    - type: f1
      value: 92.2777
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNDU0ZTQwMzg4ZDY1ZWYxOGIxMzY2ODljZTBkMTNlYjA0ODBjNjcxNTg3ZDliYWU1YTdkYTM2NTIxOTg1MGM4OCIsInZlcnNpb24iOjF9.8VHg1BXx6gLw_K7MUK2QSE80Y9guiVR8n8K8nX4laGsLibxv5u_yDv9F3ahbUa1eZG_bbidl93TY2qFUiYHtAQ
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
      value: 30.733
      name: Exact Match
    - type: f1
      value: 44.099
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
      value: 79.295
      name: Exact Match
    - type: f1
      value: 86.609
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
      value: 68.68
      name: Exact Match
    - type: f1
      value: 83.832
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
      value: 80.171
      name: Exact Match
    - type: f1
      value: 90.452
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
      value: 81.57
      name: Exact Match
    - type: f1
      value: 90.644
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
      value: 66.99
      name: Exact Match
    - type: f1
      value: 80.231
      name: F1
---

# deberta-v3-base for QA 

This is the [deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) model, fine-tuned using the [SQuAD2.0](https://huggingface.co/datasets/squad_v2) dataset. It's been trained on question-answer pairs, including unanswerable questions, for the task of Question Answering. 


## Overview
**Language model:** deberta-v3-base  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  See [an example QA pipeline on Haystack](https://haystack.deepset.ai/tutorials/first-qa-system)  
**Infrastructure**: 1x NVIDIA A10G

## Hyperparameters

```
batch_size = 12
n_epochs = 4
base_LM_model = "deberta-v3-base"
max_seq_len = 512
learning_rate = 2e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride = 128
max_query_length = 64
``` 

## Usage

### In Haystack
Haystack is an NLP framework by deepset. You can use this model in a Haystack pipeline to do question answering at scale (over many documents). To load the model in [Haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/deberta-v3-base-squad2")
# or 
reader = TransformersReader(model_name_or_path="deepset/deberta-v3-base-squad2",tokenizer="deepset/deberta-v3-base-squad2")
```

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model_name = "deepset/deberta-v3-base-squad2"
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

## Authors
**Sebastian Lee:** sebastian.lee [at] deepset.ai  
**Timo Möller:** timo.moeller [at] deepset.ai  
**Malte Pietsch:** malte.pietsch [at] deepset.ai  

## About us
<div class="grid lg:grid-cols-2 gap-x-4 gap-y-3">
    <div class="w-full h-40 object-cover mb-2 rounded-lg flex items-center justify-center">
         <img alt="" src="https://huggingface.co/spaces/deepset/README/resolve/main/haystack-logo-colored.svg" class="w-40"/>
     </div>
    <div class="w-full h-40 object-cover mb-2 rounded-lg flex items-center justify-center">
         <img alt="" src="https://huggingface.co/spaces/deepset/README/resolve/main/deepset-logo-colored.svg" class="w-40"/>
     </div>
</div>

[deepset](http://deepset.ai/) is the company behind the open-source NLP framework [Haystack](https://haystack.deepset.ai/) which is designed to help you build production ready NLP systems that use: Question answering, summarization, ranking etc.


Some of our other work: 
- [Distilled roberta-base-squad2 (aka "tinyroberta-squad2")]([https://huggingface.co/deepset/tinyroberta-squad2)
- [German BERT (aka "bert-base-german-cased")](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR datasets and models (aka "gelectra-base-germanquad", "gbert-base-germandpr")](https://deepset.ai/germanquad)

## Get in touch and join the Haystack community

<p>For more info on Haystack, visit our <strong><a href="https://github.com/deepset-ai/haystack">GitHub</a></strong> repo and <strong><a href="https://haystack.deepset.ai">Documentation</a></strong>. 

We also have a <strong><a class="h-7" href="https://haystack.deepset.ai/community/join">Discord community open to everyone!</a></strong></p>

[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai) 