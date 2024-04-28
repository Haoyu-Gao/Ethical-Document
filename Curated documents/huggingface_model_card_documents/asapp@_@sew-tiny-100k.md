---
language: en
license: apache-2.0
tags:
- speech
datasets:
- librispeech_asr
---

# SEW-tiny

[SEW by ASAPP Research](https://github.com/asappresearch/sew)

The base model pretrained on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz. Note that this model should be fine-tuned on a downstream task, like Automatic Speech Recognition, Speaker Identification, Intent Classification, Emotion Recognition, etc...

Paper: [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870)

Authors: Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi

**Abstract**
This paper is a study of performance-efficiency trade-offs in pre-trained models for automatic speech recognition (ASR). We focus on wav2vec 2.0, and formalize several architecture designs that influence both the model performance and its efficiency. Putting together all our observations, we introduce SEW (Squeezed and Efficient Wav2vec), a pre-trained model architecture with significant improvements along both performance and efficiency dimensions across a variety of training setups. For example, under the 100h-960h semi-supervised setup on LibriSpeech, SEW achieves a 1.9x inference speedup compared to wav2vec 2.0, with a 13.5% relative reduction in word error rate. With a similar inference time, SEW reduces word error rate by 25-50% across different model sizes.

The original model can be found under https://github.com/asappresearch/sew#model-checkpoints .

# Usage

See [this blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) for more information on how to fine-tune the model. Note that the class `Wav2Vec2ForCTC` has to be replaced by `SEWForCTC`.