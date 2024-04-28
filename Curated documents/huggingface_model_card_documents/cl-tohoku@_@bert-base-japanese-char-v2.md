---
language: ja
license: cc-by-sa-4.0
datasets:
- wikipedia
widget:
- text: 東北大学で[MASK]の研究をしています。
---

# BERT base Japanese (character-level tokenization with whole word masking, jawiki-20200831)

This is a [BERT](https://github.com/google-research/bert) model pretrained on texts in the Japanese language.

This version of the model processes input texts with word-level tokenization based on the Unidic 2.1.2 dictionary (available in [unidic-lite](https://pypi.org/project/unidic-lite/) package), followed by character-level tokenization.
Additionally, the model is trained with the whole word masking enabled for the masked language modeling (MLM) objective.

The codes for the pretraining are available at [cl-tohoku/bert-japanese](https://github.com/cl-tohoku/bert-japanese/tree/v2.0).

## Model architecture

The model architecture is the same as the original BERT base model; 12 layers, 768 dimensions of hidden states, and 12 attention heads.

## Training Data

The models are trained on the Japanese version of Wikipedia.
The training corpus is generated from the Wikipedia Cirrussearch dump file as of August 31, 2020.

The generated corpus files are 4.0GB in total, containing approximately 30M sentences.
We used the [MeCab](https://taku910.github.io/mecab/) morphological parser with [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) dictionary to split texts into sentences.

## Tokenization

The texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into characters.
The vocabulary size is 6144.

We used [`fugashi`](https://github.com/polm/fugashi) and [`unidic-lite`](https://github.com/polm/unidic-lite) packages for the tokenization.

## Training

The models are trained with the same configuration as the original BERT; 512 tokens per instance, 256 instances per batch, and 1M training steps.
For training of the MLM (masked language modeling) objective, we introduced whole word masking in which all of the subword tokens corresponding to a single word (tokenized by MeCab) are masked at once.

For training of each model, we used a v3-8 instance of Cloud TPUs provided by [TensorFlow Research Cloud program](https://www.tensorflow.org/tfrc/).
The training took about 5 days to finish.

## Licenses

The pretrained models are distributed under the terms of the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

## Acknowledgments

This model is trained with Cloud TPUs provided by [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc/) program.
