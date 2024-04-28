---
language:
- id
tags:
- generated_from_trainer
datasets:
- indonlp/indonlu
metrics:
- accuracy
widget:
- text: Entah mengapa saya merasakan ada sesuatu yang janggal di produk ini
model-index:
- name: roberta-base-indonesian-1.5G-sentiment-analysis-smsa
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: indonlu
      type: indonlu
      args: smsa
    metrics:
    - type: accuracy
      value: 0.9261904761904762
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-indonesian-1.5G-sentiment-analysis-smsa

This model is a fine-tuned version of [cahya/roberta-base-indonesian-1.5G](https://huggingface.co/cahya/roberta-base-indonesian-1.5G) on the indonlu dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4294
- Accuracy: 0.9262

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1500
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.6461        | 1.0   | 688  | 0.2620          | 0.9087   |
| 0.2627        | 2.0   | 1376 | 0.2291          | 0.9151   |
| 0.1784        | 3.0   | 2064 | 0.2891          | 0.9167   |
| 0.1099        | 4.0   | 2752 | 0.3317          | 0.9230   |
| 0.0857        | 5.0   | 3440 | 0.4294          | 0.9262   |
| 0.0346        | 6.0   | 4128 | 0.4759          | 0.9246   |
| 0.0221        | 7.0   | 4816 | 0.4946          | 0.9206   |
| 0.006         | 8.0   | 5504 | 0.5823          | 0.9175   |
| 0.0047        | 9.0   | 6192 | 0.5777          | 0.9159   |
| 0.004         | 10.0  | 6880 | 0.5800          | 0.9175   |


### How to use this model in Transformers Library

```python
from transformers import pipeline

pipe = pipeline(
"text-classification",
model="ayameRushia/roberta-base-indonesian-1.5G-sentiment-analysis-smsa"
)

pipe("Terima kasih atas bantuannya ya!")

```

### Framework versions

- Transformers 4.14.1
- Pytorch 1.10.0+cu111
- Datasets 1.16.1
- Tokenizers 0.10.3