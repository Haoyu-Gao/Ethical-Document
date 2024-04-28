---
language: en
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
- cc_news
---

# BigBird base model

BigBird, is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. Moreover, BigBird comes along with a theoretical understanding of the capabilities of a complete transformer that the sparse model can handle.

It is a pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in this [paper](https://arxiv.org/abs/2007.14062) and first released in this [repository](https://github.com/google-research/bigbird).

Disclaimer: The team releasing BigBird did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

BigBird relies on **block sparse attention** instead of normal attention (i.e. BERT's attention) and can handle sequences up to a length of 4096 at a much lower compute cost compared to BERT. It has achieved SOTA on various tasks involving very long sequences such as long documents summarization, question-answering with long contexts.

## How to use

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BigBirdModel

# by default its in `block_sparse` mode with num_random_blocks=3, block_size=64
model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")

# you can change `attention_type` to full attention like this:
model = BigBirdModel.from_pretrained("google/bigbird-roberta-base", attention_type="original_full")

# you can change `block_size` & `num_random_blocks` like this:
model = BigBirdModel.from_pretrained("google/bigbird-roberta-base", block_size=16, num_random_blocks=2)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

## Training Data

This model is pre-trained on four publicly available datasets: **Books**, **CC-News**, **Stories** and **Wikipedia**. It used same sentencepiece vocabulary as RoBERTa (which is in turn borrowed from GPT2).

## Training Procedure

Document longer than 4096 were split into multiple documents and documents that were much smaller than 4096 were joined. Following the original BERT training, 15% of tokens were masked and model is trained to predict the mask.

Model is warm started from RoBERTa’s checkpoint.

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
