---
license: mit
tags:
- generated_from_trainer
datasets:
- iva_mt_wslot
metrics:
- bleu
model-index:
- name: iva_mt_wslot-m2m100_418M-en-tr
  results:
  - task:
      type: text2text-generation
      name: Sequence-to-sequence Language Modeling
    dataset:
      name: iva_mt_wslot
      type: iva_mt_wslot
      config: en-tr
      split: validation
      args: en-tr
    metrics:
    - type: bleu
      value: 63.3126
      name: Bleu
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# iva_mt_wslot-m2m100_418M-en-tr

This model is a fine-tuned version of [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) on the iva_mt_wslot dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0136
- Bleu: 63.3126
- Gen Len: 19.5834

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 7
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Bleu    | Gen Len |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|:-------:|
| 0.0175        | 1.0   | 2068  | 0.0151          | 59.2285 | 19.5597 |
| 0.0121        | 2.0   | 4136  | 0.0138          | 60.2539 | 19.3643 |
| 0.0087        | 3.0   | 6204  | 0.0134          | 61.6109 | 19.3507 |
| 0.0065        | 4.0   | 8272  | 0.0134          | 61.9941 | 19.6187 |
| 0.0049        | 5.0   | 10340 | 0.0134          | 63.4822 | 19.6174 |
| 0.0039        | 6.0   | 12408 | 0.0136          | 62.9517 | 19.6493 |
| 0.0031        | 7.0   | 14476 | 0.0136          | 63.3126 | 19.5834 |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.1+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3

## Citation

If you use this model, please cite the following:
```
@article{Sowanski2023SlotLI,
  title={Slot Lost in Translation? Not Anymore: A Machine Translation Model for Virtual Assistants with Type-Independent Slot Transfer},
  author={Marcin Sowanski and Artur Janicki},
  journal={2023 30th International Conference on Systems, Signals and Image Processing (IWSSIP)},
  year={2023},
  pages={1-5}
}
```