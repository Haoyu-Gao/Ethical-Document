---
language:
- en
tags:
- speech
datasets:
- librispeech_asr
---

# UniSpeech-SAT-Base

[Microsoft's UniSpeech](https://www.microsoft.com/en-us/research/publication/unispeech-unified-speech-representation-learning-with-labeled-and-unlabeled-data/)

The base model pretrained on 16kHz sampled speech audio with utterance and speaker contrastive loss. When using the model, make sure that your speech input is also sampled at 16kHz. 

**Note**: This model does not have a tokenizer as it was pretrained on audio alone. In order to use this model **speech recognition**, a tokenizer should be created and the model should be fine-tuned on labeled text data. Check out [this blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) for more in-detail explanation of how to fine-tune the model.

The model was pre-trained on:

- 960 hours of [LibriSpeech](https://huggingface.co/datasets/librispeech_asr)

[Paper: UNISPEECH-SAT: UNIVERSAL SPEECH REPRESENTATION LEARNING WITH SPEAKER
AWARE PRE-TRAINING](https://arxiv.org/abs/2110.05752)

Authors: Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen, Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu

**Abstract**
*Self-supervised learning (SSL) is a long-standing goal for speech processing, since it utilizes large-scale unlabeled data and avoids extensive human labeling. Recent years witness great successes in applying self-supervised learning in speech recognition, while limited exploration was attempted in applying SSL for modeling speaker characteristics. In this paper, we aim to improve the existing SSL framework for speaker representation learning. Two methods are introduced for enhancing the unsupervised speaker information extraction. First, we apply the multi-task learning to the current SSL framework, where we integrate the utterance-wise contrastive loss with the SSL objective function. Second, for better speaker discrimination, we propose an utterance mixing strategy for data augmentation, where additional overlapped utterances are created unsupervisely and incorporate during training. We integrate the proposed methods into the HuBERT framework. Experiment results on SUPERB benchmark show that the proposed system achieves state-of-the-art performance in universal representation learning, especially for speaker identification oriented tasks. An ablation study is performed verifying the efficacy of each proposed method. Finally, we scale up training dataset to 94 thousand hours public audio data and achieve further performance improvement in all SUPERB tasks..*

The original model can be found under https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT.

# Usage

This is an English pre-trained speech model that has to be fine-tuned on a downstream task like speech recognition or audio classification before it can be 
used in inference. The model was pre-trained in English and should therefore perform well only in English. The model has been shown to work well on task such as speaker verification, speaker identification, and speaker diarization.

**Note**: The model was pre-trained on phonemes rather than characters. This means that one should make sure that the input text is converted to a sequence 
of phonemes before fine-tuning.

## Speech Recognition

To fine-tune the model for speech recognition, see [the official speech recognition example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/speech-recognition).

## Speech Classification

To fine-tune the model for speech classification, see [the official audio classification example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/audio-classification).

## Speaker Verification

TODO

## Speaker Diarization

TODO

# Contribution

The model was contributed by [cywang](https://huggingface.co/cywang) and [patrickvonplaten](https://huggingface.co/patrickvonplaten).

# License

The official license can be found [here](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)

![design](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/UniSpeechSAT.png)