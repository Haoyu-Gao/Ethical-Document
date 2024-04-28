---
language: ja
license: cc-by-sa-4.0
datasets:
- wikipedia
- cc100
mask_token: '[MASK]'
widget:
- text: 早稲田 大学 で 自然 言語 処理 を [MASK] する 。
---

# nlp-waseda/roberta-large-japanese-seq512

## Model description

This is a Japanese RoBERTa large model pretrained on Japanese Wikipedia and the Japanese portion of CC-100 with the maximum sequence length of 512.

## How to use

You can use this model for masked language modeling as follows:
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-large-japanese-seq512")
model = AutoModelForMaskedLM.from_pretrained("nlp-waseda/roberta-large-japanese-seq512")

sentence = '早稲田 大学 で 自然 言語 処理 を [MASK] する 。' # input should be segmented into words by Juman++ in advance
encoding = tokenizer(sentence, return_tensors='pt')
...
```

You can fine-tune this model on downstream tasks.

## Tokenization

The input text should be segmented into words by [Juman++](https://github.com/ku-nlp/jumanpp) in advance. Juman++ 2.0.0-rc3 was used for pretraining. Each word is tokenized into tokens by [sentencepiece](https://github.com/google/sentencepiece).

`BertJapaneseTokenizer` now supports automatic `JumanppTokenizer` and `SentencepieceTokenizer`. You can use [this model](https://huggingface.co/nlp-waseda/roberta-large-japanese-seq512-with-auto-jumanpp) without any data preprocessing.

## Vocabulary

The vocabulary consists of 32000 tokens including words ([JumanDIC](https://github.com/ku-nlp/JumanDIC)) and subwords induced by the unigram language model of [sentencepiece](https://github.com/google/sentencepiece).

## Training procedure

This model was trained on Japanese Wikipedia (as of 20210920) and the Japanese portion of CC-100 from the checkpoint of [nlp-waseda/roberta-large-japanese](https://huggingface.co/nlp-waseda/roberta-large-japanese). It took a week using eight NVIDIA A100 GPUs.

The following hyperparameters were used during pretraining:
- learning_rate: 6e-5
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 4120 (max_seq_length=128), 4032 (max_seq_length=512)
- max_seq_length: 512
- optimizer: Adam with betas=(0.9,0.98) and epsilon=1e-6
- lr_scheduler_type: linear
- training_steps: 670000 (max_seq_length=128) + 70000 (max_seq_length=512)
- warmup_steps: 10000
- mixed_precision_training: Native AMP
