---
language: en
license: cc-by-4.0
datasets:
- squad_v2
model-index:
- name: deepset/roberta-base-squad2
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
      value: 79.9309
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMDhhNjg5YzNiZGQ1YTIyYTAwZGUwOWEzZTRiYzdjM2QzYjA3ZTUxNDM1NjE1MTUyMjE1MGY1YzEzMjRjYzVjYiIsInZlcnNpb24iOjF9.EH5JJo8EEFwU7osPz3s7qanw_tigeCFhCXjSfyN0Y1nWVnSfulSxIk_DbAEI5iE80V4EKLyp5-mYFodWvL2KDA
    - type: f1
      value: 82.9501
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjk5ZDYwOGQyNjNkMWI0OTE4YzRmOTlkY2JjNjQ0YTZkNTMzMzNkYTA0MDFmNmI3NjA3NjNlMjhiMDQ2ZjJjNSIsInZlcnNpb24iOjF9.DDm0LNTkdLbGsue58bg1aH_s67KfbcmkvL-6ZiI2s8IoxhHJMSf29H_uV2YLyevwx900t-MwTVOW3qfFnMMEAQ
    - type: total
      value: 11869
      name: total
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMGFkMmI2ODM0NmY5NGNkNmUxYWViOWYxZDNkY2EzYWFmOWI4N2VhYzY5MGEzMTVhOTU4Zjc4YWViOGNjOWJjMCIsInZlcnNpb24iOjF9.fexrU1icJK5_MiifBtZWkeUvpmFISqBLDXSQJ8E6UnrRof-7cU0s4tX_dIsauHWtUpIHMPZCf5dlMWQKXZuAAA
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
      value: 85.289
      name: Exact Match
    - type: f1
      value: 91.841
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
      value: 29.5
      name: Exact Match
    - type: f1
      value: 40.367
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
      value: 78.567
      name: Exact Match
    - type: f1
      value: 84.469
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
      value: 69.924
      name: Exact Match
    - type: f1
      value: 83.284
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
      value: 81.204
      name: Exact Match
    - type: f1
      value: 90.595
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
      value: 82.931
      name: Exact Match
    - type: f1
      value: 90.756
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
      value: 71.55
      name: Exact Match
    - type: f1
      value: 82.939
      name: F1
---

# roberta-base for QA 

This is the [roberta-base](https://huggingface.co/roberta-base) model, fine-tuned using the [SQuAD2.0](https://huggingface.co/datasets/squad_v2) dataset. It's been trained on question-answer pairs, including unanswerable questions, for the task of Question Answering. 


## Overview
**Language model:** roberta-base  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  See [an example QA pipeline on Haystack](https://haystack.deepset.ai/tutorials/first-qa-system)  
**Infrastructure**: 4x Tesla v100

## Hyperparameters

```
batch_size = 96
n_epochs = 2
base_LM_model = "roberta-base"
max_seq_len = 386
learning_rate = 3e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
``` 

## Using a distilled model instead
Please note that we have also released a distilled version of this model called [deepset/tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2). The distilled model has a comparable prediction quality and runs at twice the speed of the base model.

## Usage

### In Haystack
Haystack is an NLP framework by deepset. You can use this model in a Haystack pipeline to do question answering at scale (over many documents). To load the model in [Haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
# or 
reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2",tokenizer="deepset/roberta-base-squad2")
```
For a complete example of ``roberta-base-squad2`` being used for  Question Answering, check out the [Tutorials in Haystack Documentation](https://haystack.deepset.ai/tutorials/first-qa-system)

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

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

## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).

```
"exact": 79.87029394424324,
"f1": 82.91251169582613,

"total": 11873,
"HasAns_exact": 77.93522267206478,
"HasAns_f1": 84.02838248389763,
"HasAns_total": 5928,
"NoAns_exact": 81.79983179142137,
"NoAns_f1": 81.79983179142137,
"NoAns_total": 5945
```

## Authors
**Branden Chan:** branden.chan@deepset.ai  
**Timo Möller:** timo.moeller@deepset.ai  
**Malte Pietsch:** malte.pietsch@deepset.ai  
**Tanay Soni:**  tanay.soni@deepset.ai 

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
