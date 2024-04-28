---
language: en
license: mit
tags:
- exbert
datasets:
- squad_v2
thumbnail: https://thumb.tildacdn.com/tild3433-3637-4830-a533-353833613061/-/resize/720x/-/format/webp/germanquad.jpg
model-index:
- name: deepset/roberta-base-squad2-distilled
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
      value: 80.8593
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzVjNzkxNmNiNDkzNzdiYjJjZGM3ZTViMGJhOGM2ZjFmYjg1MjYxMDM2YzM5NWMwNDIyYzNlN2QwNGYyNDMzZSIsInZlcnNpb24iOjF9.Rgww8tf8D7nF2dh2U_DMrFzmp87k8s7RFibrDXSvQyA66PGWXwjlsd1552lzjHnNV5hvHUM1-h3PTuY_5p64BA
    - type: f1
      value: 84.0104
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNTAyZDViNWYzNjA4OWQ5MzgyYmQ2ZDlhNWRhMTIzYTYxYzViMmI4NWE4ZGU5MzVhZTAwNTRlZmRlNWUwMjI0ZSIsInZlcnNpb24iOjF9.Er21BNgJ3jJXLuZtpubTYq9wCwO1i_VLQFwS5ET0e4eAYVVj0aOA40I5FvP5pZac3LjkCnVacxzsFWGCYVmnDA
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
      value: 86.225
      name: Exact Match
    - type: f1
      value: 92.483
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
      value: 29.9
      name: Exact Match
    - type: f1
      value: 41.183
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
      value: 79.071
      name: Exact Match
    - type: f1
      value: 84.472
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
      value: 70.733
      name: Exact Match
    - type: f1
      value: 83.958
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
      value: 82.011
      name: Exact Match
    - type: f1
      value: 91.092
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
      value: 84.203
      name: Exact Match
    - type: f1
      value: 91.521
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
      value: 72.029
      name: Exact Match
    - type: f1
      value: 83.454
      name: F1
---

## Overview
**Language model:** deepset/roberta-base-squad2-distilled   
**Language:** English  
**Training data:** SQuAD 2.0 training set
**Eval data:** SQuAD 2.0 dev set
**Infrastructure**: 4x V100 GPU  
**Published**: Dec 8th, 2021

## Details
- haystack's distillation feature was used for training. deepset/roberta-large-squad2 was used as the teacher model.

## Hyperparameters
```
batch_size = 80
n_epochs = 4
max_seq_len = 384
learning_rate = 3e-5
lr_schedule = LinearWarmup
embeds_dropout_prob = 0.1
temperature = 1.5
distillation_loss_weight = 0.75
```
## Performance
```
"exact": 79.8366040596311
"f1": 83.916407079888
```

## Authors
**Timo Möller:** timo.moeller@deepset.ai    
**Julian Risch:** julian.risch@deepset.ai    
**Malte Pietsch:** malte.pietsch@deepset.ai    
**Michel Bartels:** michel.bartels@deepset.ai    

## About us
<div class="grid lg:grid-cols-2 gap-x-4 gap-y-3">
    <div class="w-full h-40 object-cover mb-2 rounded-lg flex items-center justify-center">
         <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/deepset-logo-colored.png" class="w-40"/>
     </div>
     <div class="w-full h-40 object-cover mb-2 rounded-lg flex items-center justify-center">
         <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored.png" class="w-40"/>
     </div>
</div>

[deepset](http://deepset.ai/) is the company behind the open-source NLP framework [Haystack](https://haystack.deepset.ai/) which is designed to help you build production ready NLP systems that use: Question answering, summarization, ranking etc.


Some of our other work: 
- [Distilled roberta-base-squad2 (aka "tinyroberta-squad2")]([https://huggingface.co/deepset/tinyroberta-squad2)
- [German BERT (aka "bert-base-german-cased")](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR datasets and models (aka "gelectra-base-germanquad", "gbert-base-germandpr")](https://deepset.ai/germanquad)

## Get in touch and join the Haystack community

<p>For more info on Haystack, visit our <strong><a href="https://github.com/deepset-ai/haystack">GitHub</a></strong> repo and <strong><a href="https://docs.haystack.deepset.ai">Documentation</a></strong>. 

We also have a <strong><a class="h-7" href="https://haystack.deepset.ai/community">Discord community open to everyone!</a></strong></p>

[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](http://www.deepset.ai/jobs)