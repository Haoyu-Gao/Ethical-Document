---
language:
- multilingual
- ar
- bg
- de
- el
- en
- es
- fr
- hi
- it
- ja
- nl
- pl
- pt
- ru
- sw
- th
- tr
- ur
- vi
- zh
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
base_model: xlm-roberta-base
model-index:
- name: xlm-roberta-base-language-detection
  results: []
---

# xlm-roberta-base-language-detection

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information) dataset.

## Model description

This model is an XLM-RoBERTa transformer model with a classification head on top (i.e. a linear layer on top of the pooled output). 
For additional information please refer to the [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) model card or to the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Conneau et al.

## Intended uses & limitations

You can directly use this model as a language detector, i.e. for sequence classification tasks. Currently, it supports the following 20 languages: 

`arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)`

## Training and evaluation data

The model was fine-tuned on the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information) dataset, which consists of text sequences in 20 languages. The training set contains 70k samples, while the validation and test sets 10k each. The average accuracy on the test set is **99.6%** (this matches the average macro/weighted F1-score being the test set perfectly balanced). A more detailed evaluation is provided by the following table.

| Language | Precision | Recall | F1-score | support |
|:--------:|:---------:|:------:|:--------:|:-------:|
|ar        |0.998      |0.996   |0.997     |500      |
|bg        |0.998      |0.964   |0.981     |500      |
|de        |0.998      |0.996   |0.997     |500      |
|el        |0.996      |1.000   |0.998     |500      |
|en        |1.000      |1.000   |1.000     |500      |
|es        |0.967      |1.000   |0.983     |500      |
|fr        |1.000      |1.000   |1.000     |500      |
|hi        |0.994      |0.992   |0.993     |500      |
|it        |1.000      |0.992   |0.996     |500      |
|ja        |0.996      |0.996   |0.996     |500      |
|nl        |1.000      |1.000   |1.000     |500      |
|pl        |1.000      |1.000   |1.000     |500      |
|pt        |0.988      |1.000   |0.994     |500      |
|ru        |1.000      |0.994   |0.997     |500      |
|sw        |1.000      |1.000   |1.000     |500      |
|th        |1.000      |0.998   |0.999     |500      |
|tr        |0.994      |0.992   |0.993     |500      |
|ur        |1.000      |1.000   |1.000     |500      |
|vi        |0.992      |1.000   |0.996     |500      |
|zh        |1.000      |1.000   |1.000     |500      |

### Benchmarks

As a baseline to compare `xlm-roberta-base-language-detection` against, we have used the Python [langid](https://github.com/saffsd/langid.py) library. Since it comes pre-trained on 97 languages, we have used its `.set_languages()` method to constrain the language set to our 20 languages. The average accuracy of langid on the test set is **98.5%**. More details are provided by the table below.

| Language | Precision | Recall | F1-score | support |
|:--------:|:---------:|:------:|:--------:|:-------:|
|ar        |0.990      |0.970   |0.980     |500      |
|bg        |0.998      |0.964   |0.981     |500      |
|de        |0.992      |0.944   |0.967     |500      |
|el        |1.000      |0.998   |0.999     |500      |
|en        |1.000      |1.000   |1.000     |500      |
|es        |1.000      |0.968   |0.984     |500      |
|fr        |0.996      |1.000   |0.998     |500      |
|hi        |0.949      |0.976   |0.963     |500      |
|it        |0.990      |0.980   |0.985     |500      |
|ja        |0.927      |0.988   |0.956     |500      |
|nl        |0.980      |1.000   |0.990     |500      |
|pl        |0.986      |0.996   |0.991     |500      |
|pt        |0.950      |0.996   |0.973     |500      |
|ru        |0.996      |0.974   |0.985     |500      |
|sw        |1.000      |1.000   |1.000     |500      |
|th        |1.000      |0.996   |0.998     |500      |
|tr        |0.990      |0.968   |0.979     |500      |
|ur        |0.998      |0.996   |0.997     |500      |
|vi        |0.971      |0.990   |0.980     |500      |
|zh        |1.000      |1.000   |1.000     |500      |

## Training procedure

Fine-tuning was done via the `Trainer` API. Here is the [Colab notebook](https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing) with the training code.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 64
- eval_batch_size: 128
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2
- mixed_precision_training: Native AMP

### Training results

The validation results on the `valid` split of the Language Identification dataset are summarised here below.

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|
| 0.2492        | 1.0   | 1094 | 0.0149          | 0.9969   | 0.9969 |
| 0.0101        | 2.0   | 2188 | 0.0103          | 0.9977   | 0.9977 |

In short, it achieves the following results on the validation set:
- Loss: 0.0101
- Accuracy: 0.9977
- F1: 0.9977

### Framework versions

- Transformers 4.12.5
- Pytorch 1.10.0+cu111
- Datasets 1.15.1
- Tokenizers 0.10.3
