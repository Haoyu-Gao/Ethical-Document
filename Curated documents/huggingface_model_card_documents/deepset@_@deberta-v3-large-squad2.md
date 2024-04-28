---
language: en
license: cc-by-4.0
tags:
- deberta
- deberta-v3
- deberta-v3-large
datasets:
- squad_v2
model-index:
- name: deepset/deberta-v3-large-squad2
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
      value: 88.0876
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZmE0MWEwNjBkNTA1MmU0ZDkyYTA1OGEwNzY3NGE4NWU4NGI0NTQzNjRlNjY1NGRmNDU2MjA0NjU1N2JlZmNhYiIsInZlcnNpb24iOjF9.PnBF_vD0HujNBSShGJzsJnjmiBP_qT8xb2E7ORmpKfNspKXEuN_pBk9iV0IHRzdqOSyllcxlCv93XMPblNjWDw
    - type: f1
      value: 91.1623
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZDBkNDUzZmNkNDQwOGRkMmVlZjkxZWVlMzk3NzFmMGIxMTFmMjZlZDcyOWFiMjljNjM5MThlZDM4OWRmNzMwOCIsInZlcnNpb24iOjF9.bacyetziNI2DxO67GWpTyeRPXqF1POkyv00wEHXlyZu71pZngsNpZyrnuj2aJlCqQwHGnF_lT2ysaXKHprQRBg
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
      value: 89.2366
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjQ1Yjk3YTdiYTY1NmYxMTI1ZGZlMjRkNTlhZTkyNjRkNjgxYWJiNDk2NzE3NjAyYmY3YmRjNjg4YmEyNDkyYyIsInZlcnNpb24iOjF9.SEWyqX_FPQJOJt2KjOCNgQ2giyVeLj5bmLI5LT_Pfo33tbWPWD09TySYdsthaVTjUGT5DvDzQLASSwBH05FyBw
    - type: f1
      value: 95.0569
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2QyODQ1NWVlYjQxMjA0YTgyNmQ2NmIxOWY3MDRmZjE3ZWI5Yjc4ZDE4NzA2YjE2YTE1YTBlNzNiYmNmNzI3NCIsInZlcnNpb24iOjF9.NcXEc9xoggV76w1bQKxuJDYbOTxFzdny2k-85_b6AIMtfpYV3rGR1Z5YF6tVY2jyp7mgm5Jd5YSgGI3NvNE-CQ
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
      value: 42.1
      name: Exact Match
    - type: f1
      value: 56.587
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
      value: 83.548
      name: Exact Match
    - type: f1
      value: 89.385
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
      value: 72.979
      name: Exact Match
    - type: f1
      value: 87.254
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
      value: 83.938
      name: Exact Match
    - type: f1
      value: 92.695
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
      value: 85.534
      name: Exact Match
    - type: f1
      value: 93.153
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
      value: 73.284
      name: Exact Match
    - type: f1
      value: 85.307
      name: F1
---
# deberta-v3-large for QA 

This is the [deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) model, fine-tuned using the [SQuAD2.0](https://huggingface.co/datasets/squad_v2) dataset. It's been trained on question-answer pairs, including unanswerable questions, for the task of Question Answering. 


## Overview
**Language model:** deberta-v3-large  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  See [an example QA pipeline on Haystack](https://haystack.deepset.ai/tutorials/first-qa-system)  
**Infrastructure**: 1x NVIDIA A10G

## Hyperparameters

```
batch_size = 2
grad_acc_steps = 32
n_epochs = 6
base_LM_model = "microsoft/deberta-v3-large"
max_seq_len = 512
learning_rate = 7e-6
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
``` 

## Usage

### In Haystack
Haystack is an NLP framework by deepset. You can use this model in a Haystack pipeline to do question answering at scale (over many documents). To load the model in [Haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/deberta-v3-large-squad2")
# or 
reader = TransformersReader(model_name_or_path="deepset/deberta-v3-large-squad2",tokenizer="deepset/deberta-v3-large-squad2")
```

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/deberta-v3-large-squad2"

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
"exact": 87.6105449338836,
"f1": 90.75307008866517,

"total": 11873,
"HasAns_exact": 84.37921727395411,
"HasAns_f1": 90.6732795483674,
"HasAns_total": 5928,
"NoAns_exact": 90.83263246425568,
"NoAns_f1": 90.83263246425568,
"NoAns_total": 5945
```

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

By the way: [we're hiring!](http://www.deepset.ai/jobs) 
