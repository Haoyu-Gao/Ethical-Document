---
license: mit
tags:
- generated_from_trainer
datasets:
- common_language
metrics:
- accuracy
model-index:
- name: language-detection-fine-tuned-on-xlm-roberta-base
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: common_language
      type: common_language
      args: full
    metrics:
    - type: accuracy
      value: 0.9738386718094919
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# language-detection-fine-tuned-on-xlm-roberta-base

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the [common_language](https://huggingface.co/datasets/common_language) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1886
- Accuracy: 0.9738

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 0.1           | 1.0   | 22194 | 0.1886          | 0.9738   |


### Framework versions

- Transformers 4.12.5
- Pytorch 1.10.0+cu111
- Datasets 1.15.1
- Tokenizers 0.10.3

### Notebook
[notebook](https://github.com/IvanLauLinTiong/language-detector/blob/main/xlm_roberta_base_commonlanguage_language_detector.ipynb)