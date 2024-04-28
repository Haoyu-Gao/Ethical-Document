---
language: ko
license: apache-2.0
tags:
- automatic-speech-recognition
- generated_from_trainer
- hf-asr-leaderboard
- robust-speech-event
datasets:
- kresnik/zeroth_korean
model-index:
- name: Wav2Vec2 XLS-R 1B Korean
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: ko
    metrics:
    - type: wer
      value: 82.07
      name: Test WER
    - type: cer
      value: 42.12
      name: Test CER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Test Data
      type: speech-recognition-community-v2/eval_data
      args: ko
    metrics:
    - type: wer
      value: 82.09
      name: Test WER
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-1b](https://huggingface.co/facebook/wav2vec2-xls-r-1b) on the KRESNIK/ZEROTH_KOREAN - CLEAN dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0639
- Wer: 0.0449

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 7.5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2000
- num_epochs: 50.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| 4.603         | 0.72  | 500   | 4.6572          | 0.9985 |
| 2.6314        | 1.44  | 1000  | 2.0424          | 0.9256 |
| 2.2708        | 2.16  | 1500  | 0.9889          | 0.6989 |
| 2.1769        | 2.88  | 2000  | 0.8366          | 0.6312 |
| 2.1142        | 3.6   | 2500  | 0.7555          | 0.5998 |
| 2.0084        | 4.32  | 3000  | 0.7144          | 0.6003 |
| 1.9272        | 5.04  | 3500  | 0.6311          | 0.5461 |
| 1.8687        | 5.75  | 4000  | 0.6252          | 0.5430 |
| 1.8186        | 6.47  | 4500  | 0.5491          | 0.4988 |
| 1.7364        | 7.19  | 5000  | 0.5463          | 0.4959 |
| 1.6809        | 7.91  | 5500  | 0.4724          | 0.4484 |
| 1.641         | 8.63  | 6000  | 0.4679          | 0.4461 |
| 1.572         | 9.35  | 6500  | 0.4387          | 0.4236 |
| 1.5256        | 10.07 | 7000  | 0.3970          | 0.4003 |
| 1.5044        | 10.79 | 7500  | 0.3690          | 0.3893 |
| 1.4563        | 11.51 | 8000  | 0.3752          | 0.3875 |
| 1.394         | 12.23 | 8500  | 0.3386          | 0.3567 |
| 1.3641        | 12.95 | 9000  | 0.3290          | 0.3467 |
| 1.2878        | 13.67 | 9500  | 0.2893          | 0.3135 |
| 1.2602        | 14.39 | 10000 | 0.2723          | 0.3029 |
| 1.2302        | 15.11 | 10500 | 0.2603          | 0.2989 |
| 1.1865        | 15.83 | 11000 | 0.2440          | 0.2794 |
| 1.1491        | 16.55 | 11500 | 0.2500          | 0.2788 |
| 1.093         | 17.27 | 12000 | 0.2279          | 0.2629 |
| 1.0367        | 17.98 | 12500 | 0.2076          | 0.2443 |
| 0.9954        | 18.7  | 13000 | 0.1844          | 0.2259 |
| 0.99          | 19.42 | 13500 | 0.1794          | 0.2179 |
| 0.9385        | 20.14 | 14000 | 0.1765          | 0.2122 |
| 0.8952        | 20.86 | 14500 | 0.1706          | 0.1974 |
| 0.8841        | 21.58 | 15000 | 0.1791          | 0.1969 |
| 0.847         | 22.3  | 15500 | 0.1780          | 0.2060 |
| 0.8669        | 23.02 | 16000 | 0.1608          | 0.1862 |
| 0.8066        | 23.74 | 16500 | 0.1447          | 0.1626 |
| 0.7908        | 24.46 | 17000 | 0.1457          | 0.1655 |
| 0.7459        | 25.18 | 17500 | 0.1350          | 0.1445 |
| 0.7218        | 25.9  | 18000 | 0.1276          | 0.1421 |
| 0.703         | 26.62 | 18500 | 0.1177          | 0.1302 |
| 0.685         | 27.34 | 19000 | 0.1147          | 0.1305 |
| 0.6811        | 28.06 | 19500 | 0.1128          | 0.1244 |
| 0.6444        | 28.78 | 20000 | 0.1120          | 0.1213 |
| 0.6323        | 29.5  | 20500 | 0.1137          | 0.1166 |
| 0.5998        | 30.22 | 21000 | 0.1051          | 0.1107 |
| 0.5706        | 30.93 | 21500 | 0.1035          | 0.1037 |
| 0.5555        | 31.65 | 22000 | 0.1031          | 0.0927 |
| 0.5389        | 32.37 | 22500 | 0.0997          | 0.0900 |
| 0.5201        | 33.09 | 23000 | 0.0920          | 0.0912 |
| 0.5146        | 33.81 | 23500 | 0.0929          | 0.0947 |
| 0.515         | 34.53 | 24000 | 0.1000          | 0.0953 |
| 0.4743        | 35.25 | 24500 | 0.0922          | 0.0892 |
| 0.4707        | 35.97 | 25000 | 0.0852          | 0.0808 |
| 0.4456        | 36.69 | 25500 | 0.0855          | 0.0779 |
| 0.443         | 37.41 | 26000 | 0.0843          | 0.0738 |
| 0.4388        | 38.13 | 26500 | 0.0816          | 0.0699 |
| 0.4162        | 38.85 | 27000 | 0.0752          | 0.0645 |
| 0.3979        | 39.57 | 27500 | 0.0761          | 0.0621 |
| 0.3889        | 40.29 | 28000 | 0.0771          | 0.0625 |
| 0.3923        | 41.01 | 28500 | 0.0755          | 0.0598 |
| 0.3693        | 41.73 | 29000 | 0.0730          | 0.0578 |
| 0.3642        | 42.45 | 29500 | 0.0739          | 0.0598 |
| 0.3532        | 43.17 | 30000 | 0.0712          | 0.0553 |
| 0.3513        | 43.88 | 30500 | 0.0762          | 0.0516 |
| 0.3349        | 44.6  | 31000 | 0.0731          | 0.0504 |
| 0.3305        | 45.32 | 31500 | 0.0725          | 0.0507 |
| 0.3285        | 46.04 | 32000 | 0.0709          | 0.0489 |
| 0.3179        | 46.76 | 32500 | 0.0667          | 0.0467 |
| 0.3158        | 47.48 | 33000 | 0.0653          | 0.0494 |
| 0.3033        | 48.2  | 33500 | 0.0638          | 0.0456 |
| 0.3023        | 48.92 | 34000 | 0.0644          | 0.0464 |
| 0.2975        | 49.64 | 34500 | 0.0643          | 0.0455 |


### Framework versions

- Transformers 4.17.0.dev0
- Pytorch 1.10.2+cu102
- Datasets 1.18.3.dev0
- Tokenizers 0.11.0
