---
language: en
license: cc-by-4.0
datasets:
- squad_v2
model-index:
- name: deepset/minilm-uncased-squad2
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
      value: 76.1921
      name: Exact Match
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmViZTQ3YTBjYTc3ZDQzYmI1Mzk3MTAxM2MzNjdmMTc0MWY4Yzg2MWU3NGQ1MDJhZWI2NzY0YWYxZTY2OTgzMiIsInZlcnNpb24iOjF9.s4XCRs_pvW__LJ57dpXAEHD6NRsQ3XaFrM1xaguS6oUs5fCN77wNNc97scnfoPXT18A8RAn0cLTNivfxZm0oBA
    - type: f1
      value: 79.5483
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZmJlYTIyOTg2NjMyMzg4NzNlNGIzMTY2NDVkMjg0ODdiOWRmYjVkZDYyZjBjNWNiNTBhNjcwOWUzMDM4ZWJiZiIsInZlcnNpb24iOjF9.gxpwIBBA3_5xPi-TaZcqWNnGgCiHzxaUNgrS2jucxoVWGxhBtnPdwKVCxLleQoDDZenAXB3Yh71zMP3xTSeHCw
---

# MiniLM-L12-H384-uncased for QA

## Overview
**Language model:** microsoft/MiniLM-L12-H384-uncased  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  See an [example QA pipeline on Haystack](https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline)
**Infrastructure**: 1x Tesla v100  

## Hyperparameters

```
seed=42
batch_size = 12
n_epochs = 4
base_LM_model = "microsoft/MiniLM-L12-H384-uncased"
max_seq_len = 384
learning_rate = 4e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
grad_acc_steps=4
```

## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
```
"exact": 76.13071675229513,
"f1": 79.49786500219953,
"total": 11873,
"HasAns_exact": 78.35695006747639,
"HasAns_f1": 85.10090269418276,
"HasAns_total": 5928,
"NoAns_exact": 73.91084945332211,
"NoAns_f1": 73.91084945332211,
"NoAns_total": 5945
```

## Usage

### In Haystack
For doing QA at scale (i.e. many docs instead of single paragraph), you can load the model also in [Haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2")
# or
reader = TransformersReader(model="deepset/minilm-uncased-squad2",tokenizer="deepset/minilm-uncased-squad2")
```

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering,  AutoTokenizer, pipeline

model_name = "deepset/minilm-uncased-squad2"

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
**Vaishali Pal:** vaishali.pal@deepset.ai  
**Branden Chan:** branden.chan@deepset.ai  
**Timo Möller:** timo.moeller@deepset.ai  
**Malte Pietsch:** malte.pietsch@deepset.ai   
**Tanay Soni:** tanay.soni@deepset.ai  

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
