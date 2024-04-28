---
language:
- en
license: mit
tags:
- summarization
model-index:
- name: facebook/bart-large-xsum
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: cnn_dailymail
      type: cnn_dailymail
      config: 3.0.0
      split: test
    metrics:
    - type: rouge
      value: 25.2697
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 7.6638
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 17.1808
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 21.7933
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 3.5042972564697266
      name: loss
      verified: true
    - type: gen_len
      value: 27.4462
      name: gen_len
      verified: true
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: xsum
      type: xsum
      config: default
      split: test
    metrics:
    - type: rouge
      value: 45.4525
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 22.3455
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 37.2302
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 37.2323
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 2.3128726482391357
      name: loss
      verified: true
    - type: gen_len
      value: 25.5435
      name: gen_len
      verified: true
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: train
    metrics:
    - type: rouge
      value: 24.7852
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 5.2533
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 18.6792
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 20.629
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 3.746837854385376
      name: loss
      verified: true
    - type: gen_len
      value: 23.1206
      name: gen_len
      verified: true
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
    metrics:
    - type: rouge
      value: 24.9158
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 5.5837
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 18.8935
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 20.76
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 3.775235891342163
      name: loss
      verified: true
    - type: gen_len
      value: 23.0928
      name: gen_len
      verified: true
---
### Bart model finetuned on xsum

docs: https://huggingface.co/transformers/model_doc/bart.html

finetuning: examples/seq2seq/ (as of Aug 20, 2020)

Metrics: ROUGE > 22 on xsum.

variants: search for distilbart

paper: https://arxiv.org/abs/1910.13461