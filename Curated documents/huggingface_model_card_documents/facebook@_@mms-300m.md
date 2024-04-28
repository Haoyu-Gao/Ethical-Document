---
language:
- ab
- af
- ak
- am
- ar
- as
- av
- ay
- az
- ba
- bm
- be
- bn
- bi
- bo
- sh
- br
- bg
- ca
- cs
- ce
- cv
- ku
- cy
- da
- de
- dv
- dz
- el
- en
- eo
- et
- eu
- ee
- fo
- fa
- fj
- fi
- fr
- fy
- ff
- ga
- gl
- gn
- gu
- zh
- ht
- ha
- he
- hi
- sh
- hu
- hy
- ig
- ia
- ms
- is
- it
- jv
- ja
- kn
- ka
- kk
- kr
- km
- ki
- rw
- ky
- ko
- kv
- lo
- la
- lv
- ln
- lt
- lb
- lg
- mh
- ml
- mr
- ms
- mk
- mg
- mt
- mn
- mi
- my
- zh
- nl
- 'no'
- 'no'
- ne
- ny
- oc
- om
- or
- os
- pa
- pl
- pt
- ms
- ps
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- qu
- ro
- rn
- ru
- sg
- sk
- sl
- sm
- sn
- sd
- so
- es
- sq
- su
- sv
- sw
- ta
- tt
- te
- tg
- tl
- th
- ti
- ts
- tr
- uk
- ms
- vi
- wo
- xh
- ms
- yo
- ms
- zu
- za
license: cc-by-nc-4.0
tags:
- mms
datasets:
- google/fleurs
metrics:
- wer
---

# Massively Multilingual Speech (MMS) - 300m

Facebook's MMS counting *300m* parameters.

MMS is Facebook AI's massive multilingual pretrained model for speech ("MMS"). 
It is pretrained in with [Wav2Vec2's self-supervised training objective](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) on about 500,000 hours of speech data in over 1,400 languages.

When using the model make sure that your speech input is sampled at 16kHz. 

**Note**: This model should be fine-tuned on a downstream task, like Automatic Speech Recognition, Translation, or Classification. Check out the [**How-to-fine section](#how-to-finetune) or [**this blog**](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) for more information about ASR.

## Table Of Content

- [How to Finetune](#how-to-finetune)
- [Model details](#model-details)
- [Additional links](#additional-links)

## How to finetune

Coming soon...

## Model details

- **Developed by:** Vineel Pratap et al.
- **Model type:** Multi-Lingual Automatic Speech Recognition model
- **Language(s):** 1000+ languages
- **License:** CC-BY-NC 4.0 license
- **Num parameters**: 300 million
- **Cite as:**

      @article{pratap2023mms,
        title={Scaling Speech Technology to 1,000+ Languages},
        author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
      journal={arXiv},
      year={2023}
      }

## Additional Links

- [Blog post]( )
- [Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/mms).
- [Paper](https://arxiv.org/abs/2305.13516)
- [GitHub Repository](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr)
- [Other **MMS** checkpoints](https://huggingface.co/models?other=mms)
- MMS ASR fine-tuned checkpoints:
  - [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all)
  - [facebook/mms-1b-l1107](https://huggingface.co/facebook/mms-1b-l1107)
  - [facebook/mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102)
- [Official Space](https://huggingface.co/spaces/facebook/MMS)
