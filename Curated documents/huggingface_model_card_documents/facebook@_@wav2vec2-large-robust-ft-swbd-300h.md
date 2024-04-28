---
language: en
license: apache-2.0
tags:
- speech
- audio
- automatic-speech-recognition
datasets:
- libri_light
- common_voice
- switchboard
- fisher
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
---

# Wav2Vec2-Large-Robust finetuned on Switchboard

[Facebook's Wav2Vec2](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/).

This model is a fine-tuned version of the [wav2vec2-large-robust](https://huggingface.co/facebook/wav2vec2-large-robust) model.
It has been pretrained on:

- [Libri-Light](https://github.com/facebookresearch/libri-light): open-source audio books from the LibriVox project; clean, read-out audio data
- [CommonVoice](https://huggingface.co/datasets/common_voice): crowd-source collected audio data; read-out text snippets
- [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62): telephone speech corpus; noisy telephone data
- [Fisher](https://catalog.ldc.upenn.edu/LDC2004T19): conversational telephone speech; noisy telephone data

and subsequently been finetuned on 300 hours of

- [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62): telephone speech corpus; noisy telephone data

When using the model make sure that your speech input is also sampled at 16Khz. 

[Paper Robust Wav2Vec2](https://arxiv.org/abs/2104.01027)

Authors: Wei-Ning Hsu, Anuroop Sriram, Alexei Baevski, Tatiana Likhomanenko, Qiantong Xu, Vineel Pratap, Jacob Kahn, Ann Lee, Ronan Collobert, Gabriel Synnaeve, Michael Auli

**Abstract**
Self-supervised learning of speech representations has been a very active research area but most work is focused on a single domain such as read audio books for which there exist large quantities of labeled and unlabeled data. In this paper, we explore more general setups where the domain of the unlabeled data for pre-training data differs from the domain of the labeled data for fine-tuning, which in turn may differ from the test data domain. Our experiments show that using target domain data during pre-training leads to large performance improvements across a variety of setups. On a large-scale competitive setup, we show that pre-training on unlabeled in-domain data reduces the gap between models trained on in-domain and out-of-domain labeled data by 66%-73%. This has obvious practical implications since it is much easier to obtain unlabeled target domain data than labeled data. Moreover, we find that pre-training on multiple domains improves generalization performance on domains not seen during training. Code and models will be made available at this https URL.

The original model can be found under https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20.

# Usage

To transcribe audio files the model can be used as a standalone acoustic model as follows:

```python
 from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
 from datasets import load_dataset
 import torch
 
 # load model and processor
 processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
 model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
     
 # load dummy dataset and read soundfiles
 ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 
 # tokenize
 input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
 
 # retrieve logits
 logits = model(input_values).logits
 
 # take argmax and decode
 predicted_ids = torch.argmax(logits, dim=-1)
 transcription = processor.batch_decode(predicted_ids)
 ```