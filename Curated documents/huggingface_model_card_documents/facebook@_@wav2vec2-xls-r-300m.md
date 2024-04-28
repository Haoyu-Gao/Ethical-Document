---
language:
- multilingual
- ab
- af
- sq
- am
- ar
- hy
- as
- az
- ba
- eu
- be
- bn
- bs
- br
- bg
- my
- yue
- ca
- ceb
- km
- zh
- cv
- hr
- cs
- da
- dv
- nl
- en
- eo
- et
- fo
- fi
- fr
- gl
- lg
- ka
- de
- el
- gn
- gu
- ht
- cnh
- ha
- haw
- he
- hi
- hu
- is
- id
- ia
- ga
- it
- ja
- jv
- kb
- kn
- kk
- rw
- ky
- ko
- ku
- lo
- la
- lv
- ln
- lt
- lm
- mk
- mg
- ms
- ml
- mt
- gv
- mi
- mr
- mn
- ne
- false
- nn
- oc
- or
- ps
- fa
- pl
- pt
- pa
- ro
- rm
- rm
- ru
- sah
- sa
- sco
- sr
- sn
- sd
- si
- sk
- sl
- so
- hsb
- es
- su
- sw
- sv
- tl
- tg
- ta
- tt
- te
- th
- bo
- tp
- tr
- tk
- uk
- ur
- uz
- vi
- vot
- war
- cy
- yi
- yo
- zu
license: apache-2.0
tags:
- speech
- xls_r
- xls_r_pretrained
datasets:
- common_voice
- multilingual_librispeech
language_bcp47:
- zh-HK
- zh-TW
- fy-NL
---

# Wav2Vec2-XLS-R-300M

[Facebook's Wav2Vec2 XLS-R](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) counting **300 million** parameters.

![model image](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/xls_r.png)

XLS-R is Facebook AI's large-scale multilingual pretrained model for speech (the "XLM-R for Speech"). It is pretrained on 436k hours of unlabeled speech, including VoxPopuli, MLS, CommonVoice, BABEL, and VoxLingua107. It uses the wav2vec 2.0 objective, in 128 languages. When using the model make sure that your speech input is sampled at 16kHz. 

**Note**: This model should be fine-tuned on a downstream task, like Automatic Speech Recognition, Translation, or Classification. Check out [**this blog**](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) for more information about ASR.

[XLS-R Paper](https://arxiv.org/abs/2111.09296)

Authors: Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli

**Abstract**
This paper presents XLS-R, a large-scale model for cross-lingual speech representation learning based on wav2vec 2.0. We train models with up to 2B parameters on 436K hours of publicly available speech audio in 128 languages, an order of magnitude more public data than the largest known prior work. Our evaluation covers a wide range of tasks, domains, data regimes and languages, both high and low-resource. On the CoVoST-2 speech translation benchmark, we improve the previous state of the art by an average of 7.4 BLEU over 21 translation directions into English. For speech recognition, XLS-R improves over the best known prior work on BABEL, MLS, CommonVoice as well as VoxPopuli, lowering error rates by 20%-33% relative on average. XLS-R also sets a new state of the art on VoxLingua107 language identification. Moreover, we show that with sufficient model size, cross-lingual pretraining can outperform English-only pretraining when translating English speech into other languages, a setting which favors monolingual pretraining. We hope XLS-R can help to improve speech processing tasks for many more languages of the world.

The original model can be found under https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20.

# Usage

See [this google colab](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLS_R_on_Common_Voice.ipynb) for more information on how to fine-tune the model.

You can find other pretrained XLS-R models with different numbers of parameters:

* [300M parameters version](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* [1B version version](https://huggingface.co/facebook/wav2vec2-xls-r-1b)
* [2B version version](https://huggingface.co/facebook/wav2vec2-xls-r-2b)