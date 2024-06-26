---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- xsum
model-index:
- name: t5-small-xsum
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5-small-xsum

This model is a fine-tuned version of [t5-small](https://huggingface.co/t5-small) on the xsum dataset.
It achieves the following results on the evaluation set:
- Loss: 2.3953

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step  | Validation Loss |
|:-------------:|:-----:|:-----:|:---------------:|
| 2.8641        | 0.04  | 500   | 2.6202          |
| 2.7466        | 0.08  | 1000  | 2.5660          |
| 2.8767        | 0.12  | 1500  | 2.5319          |
| 2.7099        | 0.16  | 2000  | 2.5107          |
| 2.7752        | 0.2   | 2500  | 2.4922          |
| 2.6037        | 0.24  | 3000  | 2.4800          |
| 2.8236        | 0.27  | 3500  | 2.4677          |
| 2.7089        | 0.31  | 4000  | 2.4581          |
| 2.7299        | 0.35  | 4500  | 2.4498          |
| 2.7498        | 0.39  | 5000  | 2.4420          |
| 2.6186        | 0.43  | 5500  | 2.4346          |
| 2.7817        | 0.47  | 6000  | 2.4288          |
| 2.5559        | 0.51  | 6500  | 2.4239          |
| 2.6725        | 0.55  | 7000  | 2.4186          |
| 2.6316        | 0.59  | 7500  | 2.4149          |
| 2.5561        | 0.63  | 8000  | 2.4115          |
| 2.5708        | 0.67  | 8500  | 2.4097          |
| 2.5861        | 0.71  | 9000  | 2.4052          |
| 2.6363        | 0.74  | 9500  | 2.4024          |
| 2.7435        | 0.78  | 10000 | 2.4003          |
| 2.7258        | 0.82  | 10500 | 2.3992          |
| 2.6113        | 0.86  | 11000 | 2.3983          |
| 2.6006        | 0.9   | 11500 | 2.3972          |
| 2.5684        | 0.94  | 12000 | 2.3960          |
| 2.6181        | 0.98  | 12500 | 2.3953          |


### Framework versions

- Transformers 4.18.0
- Pytorch 1.10.0+cu111
- Datasets 2.0.0
- Tokenizers 0.11.6
