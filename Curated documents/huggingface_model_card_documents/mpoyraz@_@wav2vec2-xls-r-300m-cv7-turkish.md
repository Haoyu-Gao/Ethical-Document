---
language: tr
license: cc-by-4.0
tags:
- automatic-speech-recognition
- hf-asr-leaderboard
- mozilla-foundation/common_voice_7_0
- robust-speech-event
- tr
datasets:
- mozilla-foundation/common_voice_7_0
model-index:
- name: mpoyraz/wav2vec2-xls-r-300m-cv7-turkish
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice 7
      type: mozilla-foundation/common_voice_7_0
      args: tr
    metrics:
    - type: wer
      value: 8.62
      name: Test WER
    - type: cer
      value: 2.26
      name: Test CER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: tr
    metrics:
    - type: wer
      value: 30.87
      name: Test WER
    - type: cer
      value: 10.69
      name: Test CER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Test Data
      type: speech-recognition-community-v2/eval_data
      args: tr
    metrics:
    - type: wer
      value: 32.09
      name: Test WER
---


# wav2vec2-xls-r-300m-cv7-turkish

## Model description
This ASR model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on Turkish language.

## Training and evaluation data
The following datasets were used for finetuning:
 - [Common Voice 7.0 TR](https://huggingface.co/datasets/mozilla-foundation/common_voice_7_0) All `validated` split except `test` split was used for training.
 - [MediaSpeech](https://www.openslr.org/108/)

## Training procedure
To support both of the datasets above, custom pre-processing and loading steps was performed and [wav2vec2-turkish](https://github.com/mpoyraz/wav2vec2-turkish) repo was used for that purpose.

### Training hyperparameters
The following hypermaters were used for finetuning:
- learning_rate 2e-4
- num_train_epochs 10
- warmup_steps 500
- freeze_feature_extractor
- mask_time_prob 0.1
- mask_feature_prob 0.05
- feat_proj_dropout 0.05
- attention_dropout 0.05
- final_dropout 0.05
- activation_dropout 0.05
- per_device_train_batch_size 8
- per_device_eval_batch_size 8
- gradient_accumulation_steps 8

### Framework versions
- Transformers 4.16.0.dev0
- Pytorch 1.10.1
- Datasets 1.17.0
- Tokenizers 0.10.3

## Language Model
N-gram language model is trained on a Turkish Wikipedia articles using KenLM and [ngram-lm-wiki](https://github.com/mpoyraz/ngram-lm-wiki) repo was used to generate arpa LM and convert it into binary format.

## Evaluation Commands
Please install [unicode_tr](https://pypi.org/project/unicode_tr/) package before running evaluation. It is used for Turkish text processing.
1. To evaluate on `mozilla-foundation/common_voice_7_0` with split `test`
```bash
python eval.py --model_id mpoyraz/wav2vec2-xls-r-300m-cv7-turkish --dataset mozilla-foundation/common_voice_7_0 --config tr --split test
```

2. To evaluate on `speech-recognition-community-v2/dev_data`

```bash
python eval.py --model_id mpoyraz/wav2vec2-xls-r-300m-cv7-turkish --dataset speech-recognition-community-v2/dev_data --config tr --split validation --chunk_length_s 5.0 --stride_length_s 1.0
```
## Evaluation results:

| Dataset | WER | CER |
|---|---|---|
|Common Voice 7 TR test split| 8.62 | 2.26 |
|Speech Recognition Community dev data| 30.87 | 10.69 |
