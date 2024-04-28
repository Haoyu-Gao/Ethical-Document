---
language: en
license: apache-2.0
tags:
- summarization
datasets:
- scientific_papers
model-index:
- name: google/bigbird-pegasus-large-arxiv
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: scientific_papers
      type: scientific_papers
      config: pubmed
      split: test
    metrics:
    - type: rouge
      value: 36.0276
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 13.4166
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 21.9612
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 29.648
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 2.774355173110962
      name: loss
      verified: true
    - type: meteor
      value: 0.2824
      name: meteor
      verified: true
    - type: gen_len
      value: 209.2537
      name: gen_len
      verified: true
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
      value: 9.0885
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 1.0325
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 7.3182
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 8.1455
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: .nan
      name: loss
      verified: true
    - type: gen_len
      value: 210.4762
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
      value: 4.9787
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 0.3527
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 4.3679
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 4.1723
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: .nan
      name: loss
      verified: true
    - type: gen_len
      value: 230.4886
      name: gen_len
      verified: true
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: scientific_papers
      type: scientific_papers
      config: arxiv
      split: test
    metrics:
    - type: rouge
      value: 43.4702
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 17.4297
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 26.2587
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 35.5587
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 2.1113228797912598
      name: loss
      verified: true
    - type: gen_len
      value: 183.3702
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
      value: 3.621
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 0.1699
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 3.2016
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 3.3269
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 7.664482116699219
      name: loss
      verified: true
    - type: gen_len
      value: 233.8107
      name: gen_len
      verified: true
---

# BigBirdPegasus model (large)

BigBird, is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. Moreover, BigBird comes along with a theoretical understanding of the capabilities of a complete transformer that the sparse model can handle. 

BigBird was introduced in this [paper](https://arxiv.org/abs/2007.14062) and first released in this [repository](https://github.com/google-research/bigbird).

Disclaimer: The team releasing BigBird did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

BigBird relies on **block sparse attention** instead of normal attention (i.e. BERT's attention) and can handle sequences up to a length of 4096 at a much lower compute cost compared to BERT. It has achieved SOTA on various tasks involving very long sequences such as long documents summarization, question-answering with long contexts.

## How to use

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

# by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")

# decoder attention type can't be changed & will be "original_full"
# you can change `attention_type` (encoder only) to full attention like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv", attention_type="original_full")

# you can change `block_size` & `num_random_blocks` like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv", block_size=16, num_random_blocks=2)

text = "Replace me by any text you'd like."
inputs = tokenizer(text, return_tensors='pt')
prediction = model.generate(**inputs)
prediction = tokenizer.batch_decode(prediction)
```

## Training Procedure

This checkpoint is obtained after fine-tuning `BigBirdPegasusForConditionalGeneration` for **summarization** on **arxiv dataset** from [scientific_papers](https://huggingface.co/datasets/scientific_papers).

## BibTeX entry and citation info

```tex
@misc{zaheer2021big,
      title={Big Bird: Transformers for Longer Sequences}, 
      author={Manzil Zaheer and Guru Guruganesh and Avinava Dubey and Joshua Ainslie and Chris Alberti and Santiago Ontanon and Philip Pham and Anirudh Ravula and Qifan Wang and Li Yang and Amr Ahmed},
      year={2021},
      eprint={2007.14062},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
