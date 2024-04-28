---
language: en
license: apache-2.0
tags:
- speech
datasets:
- librispeech_asr
---

# Data2Vec-Audio-Large

[Facebook's Data2Vec](https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/)

The large model pretrained on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz. 

**Note**: This model does not have a tokenizer as it was pretrained on audio alone. In order to use this model **speech recognition**, a tokenizer should be created and the model should be fine-tuned on labeled text data. Check out [this blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) for more in-detail explanation of how to fine-tune the model.

[Paper](https://arxiv.org/abs/2202.03555)

Authors: Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli

**Abstract**

While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.

The original model can be found under https://github.com/pytorch/fairseq/tree/main/examples/data2vec .

# Pre-Training method

![model image](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/data2vec.png)

For more information, please take a look at the [official paper](https://arxiv.org/abs/2202.03555).

# Usage

See [this notebook](https://colab.research.google.com/drive/1FjTsqbYKphl9kL-eILgUc-bl4zVThL8F?usp=sharing) for more information on how to fine-tune the model.