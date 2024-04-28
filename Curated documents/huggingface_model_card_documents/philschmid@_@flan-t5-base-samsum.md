---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- samsum
metrics:
- rouge
model-index:
- name: flan-t5-base-samsum
  results:
  - task:
      type: text2text-generation
      name: Sequence-to-sequence Language Modeling
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: train
      args: samsum
    metrics:
    - type: rouge
      value: 47.2358
      name: Rouge1
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-base-samsum

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the samsum dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3716
- Rouge1: 47.2358
- Rouge2: 23.5135
- Rougel: 39.6266
- Rougelsum: 43.3458
- Gen Len: 17.3907

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 1.4379        | 1.0   | 1842 | 1.3805          | 47.1075 | 23.531  | 39.6919 | 43.549    | 17.1197 |
| 1.3559        | 2.0   | 3684 | 1.3716          | 47.2358 | 23.5135 | 39.6266 | 43.3458   | 17.3907 |
| 1.2783        | 3.0   | 5526 | 1.3721          | 47.4581 | 23.7339 | 39.7726 | 43.4568   | 17.1832 |
| 1.2378        | 4.0   | 7368 | 1.3757          | 47.8557 | 24.0593 | 40.2324 | 44.0085   | 17.3053 |
| 1.1983        | 5.0   | 9210 | 1.3751          | 47.8156 | 24.0038 | 40.2169 | 43.8918   | 17.3040 |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.12.1+cu113
- Datasets 2.8.0
- Tokenizers 0.12.1
