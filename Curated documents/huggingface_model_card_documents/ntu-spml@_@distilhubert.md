---
language: en
license: apache-2.0
tags:
- speech
datasets:
- librispeech_asr
---

# DistilHuBERT

[DistilHuBERT by NTU Speech Processing & Machine Learning Lab](https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)

The base model pretrained on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz.

**Note**: This model does not have a tokenizer as it was pretrained on audio alone. In order to use this model **speech recognition**, a tokenizer should be created and the model should be fine-tuned on labeled text data. Check out [this blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) for more in-detail explanation of how to fine-tune the model.

Paper: [DistilHuBERT: Speech Representation Learning by Layer-wise Distillation of Hidden-unit BERT](https://arxiv.org/abs/2110.01900)

Authors: Heng-Jui Chang, Shu-wen Yang, Hung-yi Lee

**Abstract**
Self-supervised speech representation learning methods like wav2vec 2.0 and Hidden-unit BERT (HuBERT) leverage unlabeled speech data for pre-training and offer good representations for numerous speech processing tasks. Despite the success of these methods, they require large memory and high pre-training costs, making them inaccessible for researchers in academia and small companies. Therefore, this paper introduces DistilHuBERT, a novel multi-task learning framework to distill hidden representations from a HuBERT model directly. This method reduces HuBERT's size by 75% and 73% faster while retaining most performance in ten different tasks. Moreover, DistilHuBERT required little training time and data, opening the possibilities of pre-training personal and on-device SSL models for speech.

The original model can be found under https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller .

# Usage

See [this blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) for more information on how to fine-tune the model. Note that the class `Wav2Vec2ForCTC` has to be replaced by `HubertForCTC`.
