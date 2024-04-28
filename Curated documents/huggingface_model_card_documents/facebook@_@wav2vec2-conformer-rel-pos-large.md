---
language: en
license: apache-2.0
tags:
- speech
datasets:
- librispeech_asr
---

# Wav2Vec2-Conformer-Large with Relative Position Embeddings

Wav2Vec2 Conformer with relative position embeddings, pretrained on 960 hours of Librispeech on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz.

**Note**: This model does not have a tokenizer as it was pretrained on audio alone. In order to use this model **speech recognition**, a tokenizer should be created and the model should be fine-tuned on labeled text data. Check out [this blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) for more in-detail explanation of how to fine-tune the model.

**Paper**: [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171)

**Authors**: Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino

The results of Wav2Vec2-Conformer can be found in Table 3 and Table 4 of the [official paper](https://arxiv.org/abs/2010.05171).

The original model can be found under https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20.

# Usage

See [this notebook](https://colab.research.google.com/drive/1FjTsqbYKphl9kL-eILgUc-bl4zVThL8F?usp=sharing) for more information on how to fine-tune the model.