---
language:
- pt
license: apache-2.0
tags:
- automatic-speech-recognition
- generated_from_trainer
- robust-speech-event
- pt
- hf-asr-leaderboard
datasets:
- common_voice
model-index:
- name: wav2vec2-large-xls-r-300m-pt-cv
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice 6
      type: common_voice
      args: pt
    metrics:
    - type: wer
      value: 24.29
      name: Test WER
    - type: cer
      value: 7.51
      name: Test CER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: sv
    metrics:
    - type: wer
      value: 55.72
      name: Test WER
    - type: cer
      value: 21.82
      name: Test CER
    - type: wer
      value: 47.88
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Test Data
      type: speech-recognition-community-v2/eval_data
      args: pt
    metrics:
    - type: wer
      value: 50.78
      name: Test WER
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-large-xls-r-300m-pt-cv

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the common_voice dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3418
- Wer: 0.3581

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 10.9035       | 0.2   | 100  | 4.2750          | 1.0    |
| 3.3275        | 0.41  | 200  | 3.0334          | 1.0    |
| 3.0016        | 0.61  | 300  | 2.9494          | 1.0    |
| 2.1874        | 0.82  | 400  | 1.4355          | 0.8721 |
| 1.09          | 1.02  | 500  | 0.9987          | 0.7165 |
| 0.8251        | 1.22  | 600  | 0.7886          | 0.6406 |
| 0.6927        | 1.43  | 700  | 0.6753          | 0.5801 |
| 0.6143        | 1.63  | 800  | 0.6300          | 0.5509 |
| 0.5451        | 1.84  | 900  | 0.5586          | 0.5156 |
| 0.5003        | 2.04  | 1000 | 0.5493          | 0.5027 |
| 0.3712        | 2.24  | 1100 | 0.5271          | 0.4872 |
| 0.3486        | 2.45  | 1200 | 0.4953          | 0.4817 |
| 0.3498        | 2.65  | 1300 | 0.4619          | 0.4538 |
| 0.3112        | 2.86  | 1400 | 0.4570          | 0.4387 |
| 0.3013        | 3.06  | 1500 | 0.4437          | 0.4147 |
| 0.2136        | 3.27  | 1600 | 0.4176          | 0.4124 |
| 0.2131        | 3.47  | 1700 | 0.4281          | 0.4194 |
| 0.2099        | 3.67  | 1800 | 0.3864          | 0.3949 |
| 0.1925        | 3.88  | 1900 | 0.3926          | 0.3913 |
| 0.1709        | 4.08  | 2000 | 0.3764          | 0.3804 |
| 0.1406        | 4.29  | 2100 | 0.3787          | 0.3742 |
| 0.1342        | 4.49  | 2200 | 0.3645          | 0.3693 |
| 0.1305        | 4.69  | 2300 | 0.3463          | 0.3625 |
| 0.1298        | 4.9   | 2400 | 0.3418          | 0.3581 |


### Framework versions

- Transformers 4.11.3
- Pytorch 1.10.0+cu111
- Datasets 1.13.3
- Tokenizers 0.10.3
