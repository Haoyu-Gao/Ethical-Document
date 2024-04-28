---
language:
- multilingual
datasets:
- squad
- arcd
- xquad
---

# Multilingual BERT fine-tuned on SQuADv1.1

[**WandB run link**](https://wandb.ai/salti/mBERT_QA/runs/wkqzhrp2)

**GPU**: Tesla P100-PCIE-16GB

## Training Arguments

```python
max_seq_length              = 512
doc_stride                  = 256
max_answer_length           = 64
bacth_size                  = 16
gradient_accumulation_steps = 2
learning_rate               = 5e-5
weight_decay                = 3e-7
num_train_epochs            = 3
warmup_ratio                = 0.1
fp16                        = True
fp16_opt_level              = "O1"
seed                        = 0
```

## Results

|   EM   |   F1   |
| :----: | :----: |
| 81.731 | 89.009 |

## Zero-shot performance

### on ARCD

|   EM   |   F1   |
| :----: | :----: |
| 20.655 | 48.051 |

### on XQuAD

|  Language  |   EM   |   F1   |
| :--------: | :----: | :----: |
|   Arabic   | 42.185 | 57.803 |
|  English   | 73.529 | 85.01  |
|   German   | 55.882 | 72.555 |
|   Greek    | 45.21  | 62.207 |
|  Spanish   | 58.067 | 76.406 |
|   Hindi    | 40.588 | 55.29  |
|  Russian   | 55.126 | 71.617 |
|    Thai    | 26.891 | 39.965 |
|  Turkish   | 34.874 | 51.138 |
| Vietnamese | 47.983 | 68.125 |
|  Chinese   | 47.395 | 58.928 |
