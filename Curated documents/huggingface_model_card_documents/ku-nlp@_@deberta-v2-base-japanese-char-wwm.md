---
language: ja
license: cc-by-sa-4.0
library_name: transformers
tags:
- deberta
- deberta-v2
- fill-mask
- character
- wwm
datasets:
- wikipedia
- cc100
- oscar
metrics:
- accuracy
mask_token: '[MASK]'
widget:
- text: 京都大学で自然言語処理を[MASK][MASK]する。
---

# Model Card for Japanese character-level DeBERTa V2 base

## Model description

This is a Japanese DeBERTa V2 base model pre-trained on Japanese Wikipedia, the Japanese portion of CC-100, and the Japanese portion of OSCAR.
This model is trained with character-level tokenization and whole word masking.

## How to use

You can use this model for masked language modeling as follows:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('ku-nlp/deberta-v2-base-japanese-char-wwm')
model = AutoModelForMaskedLM.from_pretrained('ku-nlp/deberta-v2-base-japanese-char-wwm')

sentence = '京都大学で自然言語処理を[MASK][MASK]する。'
encoding = tokenizer(sentence, return_tensors='pt')
...
```

You can also fine-tune this model on downstream tasks.

## Tokenization

There is no need to tokenize texts in advance, and you can give raw texts to the tokenizer.
The texts are tokenized into character-level tokens by [sentencepiece](https://github.com/google/sentencepiece).

## Training data

We used the following corpora for pre-training:

- Japanese Wikipedia (as of 20221020, 3.2GB, 27M sentences, 1.3M documents)
- Japanese portion of CC-100 (85GB, 619M sentences, 66M documents)
- Japanese portion of OSCAR (54GB, 326M sentences, 25M documents)

Note that we filtered out documents annotated with "header", "footer", or "noisy" tags in OSCAR.
Also note that Japanese Wikipedia was duplicated 10 times to make the total size of the corpus comparable to that of CC-100 and OSCAR. As a result, the total size of the training data is 171GB.

## Training procedure

We first segmented texts in the corpora into words using [Juman++ 2.0.0-rc3](https://github.com/ku-nlp/jumanpp/releases/tag/v2.0.0-rc3) for whole word masking.
Then, we built a sentencepiece model with 22,012 tokens including all characters that appear in the training corpus.

We tokenized raw corpora into character-level subwords using the sentencepiece model and trained the Japanese DeBERTa model using [transformers](https://github.com/huggingface/transformers) library.
The training took 20 days using 8 NVIDIA A100-SXM4-40GB GPUs.

The following hyperparameters were used during pre-training:

- learning_rate: 2e-4
- per_device_train_batch_size: 46
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 6
- total_train_batch_size: 2,208
- max_seq_length: 512
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-06
- lr_scheduler_type: linear schedule with warmup (lr = 0 at 500k steps)
- training_steps: 320,000
- warmup_steps: 10,000

## Acknowledgments

This work was supported by Joint Usage/Research Center for Interdisciplinary Large-scale Information Infrastructures (JHPCN) through General Collaboration Project no. jh221004, "Developing a Platform for Constructing and Sharing of Large-Scale Japanese Language Models".
For training models, we used the mdx: a platform for the data-driven future.
