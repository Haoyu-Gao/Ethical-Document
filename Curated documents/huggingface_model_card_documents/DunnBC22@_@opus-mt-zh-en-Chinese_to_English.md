---
language:
- en
- zh
license: cc-by-4.0
tags:
- generated_from_trainer
datasets:
- GEM/wiki_lingua
metrics:
- bleu
- rouge
base_model: Helsinki-NLP/opus-mt-zh-en
pipeline_tag: translation
model-index:
- name: opus-mt-zh-en-Chinese_to_English
  results: []
---

# opus-mt-zh-en-Chinese_to_English

This model is a fine-tuned version of [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en).

## Model description

For more information on how it was created, check out the following link: https://github.com/DunnBC22/NLP_Projects/blob/main/Machine%20Translation/Chinese%20to%20English%20Translation/Chinese_to_English_Translation.ipynb

## Intended uses & limitations

This model is intended to demonstrate my ability to solve a complex problem using technology.

## Training and evaluation data

Dataset Source: https://huggingface.co/datasets/GEM/wiki_lingua

__Chinese Text Length__
![Chinese Text Length](https://raw.githubusercontent.com/DunnBC22/NLP_Projects/main/Machine%20Translation/Chinese%20to%20English%20Translation/Images/Histogram%20-%20Chinese%20Text%20Length.png)

__English Text Length__
![English Text Length__](https://raw.githubusercontent.com/DunnBC22/NLP_Projects/main/Machine%20Translation/Chinese%20to%20English%20Translation/Images/Histogram%20-%20English%20Text%20Length.png)

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results


| Epoch | Validation Loss | Bleu | Rouge1 | Rouge2 | RougeL | RougeLsum | Avg. Prediction Lengths |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1.0 | 1.0113 | 45.2808 | 0.6201 | 0.4198 | 0.5927 | 0.5927 | 24.5581 |

### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1+cu118
- Datasets 2.14.4
- Tokenizers 0.13.3