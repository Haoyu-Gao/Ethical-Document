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
- acc
---

# Massively Multilingual Speech (MMS) - Finetuned LID

This checkpoint is a model fine-tuned for speech language identification (LID) and part of Facebook's [Massive Multilingual Speech project](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/).
This checkpoint is based on the [Wav2Vec2 architecture](https://huggingface.co/docs/transformers/model_doc/wav2vec2) and classifies raw audio input to a probability distribution over 126 output classes (each class representing a language).
The checkpoint consists of **1 billion parameters** and has been fine-tuned from [facebook/mms-1b](https://huggingface.co/facebook/mms-1b) on 126 languages.

## Table Of Content

- [Example](#example)
- [Supported Languages](#supported-languages)
- [Model details](#model-details)
- [Additional links](#additional-links)

## Example

This MMS checkpoint can be used with [Transformers](https://github.com/huggingface/transformers) to identify
the spoken language of an audio. It can recognize the [following 126 languages](#supported-languages).

Let's look at a simple example.

First, we install transformers and some other libraries
```
pip install torch accelerate torchaudio datasets
pip install --upgrade transformers
````

**Note**: In order to use MMS you need to have at least `transformers >= 4.30` installed. If the `4.30` version
is not yet available [on PyPI](https://pypi.org/project/transformers/) make sure to install `transformers` from 
source:
```
pip install git+https://github.com/huggingface/transformers.git
```

Next, we load a couple of audio samples via `datasets`. Make sure that the audio data is sampled to 16000 kHz.

```py
from datasets import load_dataset, Audio

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# Arabic
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
ar_sample = next(iter(stream_data))["audio"]["array"]
```

Next, we load the model and processor

```py
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-126"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
```

Now we process the audio data, pass the processed audio data to the model to classify it into a language, just like we usually do for Wav2Vec2 audio classification models such as [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition)

```py
# English
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'eng'

# Arabic
inputs = processor(ar_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'ara'
```

To see all the supported languages of a checkpoint, you can print out the language ids as follows:
```py
processor.id2label.values()
```

For more details, about the architecture please have a look at [the official docs](https://huggingface.co/docs/transformers/main/en/model_doc/mms).

## Supported Languages

This model supports 126 languages. Unclick the following to toogle all supported languages of this checkpoint in [ISO 639-3 code](https://en.wikipedia.org/wiki/ISO_639-3).
You can find more details about the languages and their ISO 649-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html).
<details>
  <summary>Click to toggle</summary>

- ara
- cmn
- eng
- spa
- fra
- mlg
- swe
- por
- vie
- ful
- sun
- asm
- ben
- zlm
- kor
- ind
- hin
- tuk
- urd
- aze
- slv
- mon
- hau
- tel
- swh
- bod
- rus
- tur
- heb
- mar
- som
- tgl
- tat
- tha
- cat
- ron
- mal
- bel
- pol
- yor
- nld
- bul
- hat
- afr
- isl
- amh
- tam
- hun
- hrv
- lit
- cym
- fas
- mkd
- ell
- bos
- deu
- sqi
- jav
- nob
- uzb
- snd
- lat
- nya
- grn
- mya
- orm
- lin
- hye
- yue
- pan
- jpn
- kaz
- npi
- kat
- guj
- kan
- tgk
- ukr
- ces
- lav
- bak
- khm
- fao
- glg
- ltz
- lao
- mlt
- sin
- sna
- ita
- srp
- mri
- nno
- pus
- eus
- ory
- lug
- bre
- luo
- slk
- fin
- dan
- yid
- est
- ceb
- war
- san
- kir
- oci
- wol
- haw
- kam
- umb
- xho
- epo
- zul
- ibo
- abk
- ckb
- nso
- gle
- kea
- ast
- sco
- glv
- ina

</details>

## Model details

- **Developed by:** Vineel Pratap et al.
- **Model type:** Multi-Lingual Automatic Speech Recognition model
- **Language(s):** 126 languages, see [supported languages](#supported-languages)
- **License:** CC-BY-NC 4.0 license
- **Num parameters**: 1 billion
- **Audio sampling rate**: 16,000 kHz
- **Cite as:**

      @article{pratap2023mms,
        title={Scaling Speech Technology to 1,000+ Languages},
        author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
      journal={arXiv},
      year={2023}
      }

## Additional Links

- [Blog post](https://ai.facebook.com/blog/multilingual-model-speech-recognition/)
- [Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/mms).
- [Paper](https://arxiv.org/abs/2305.13516)
- [GitHub Repository](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr)
- [Other **MMS** checkpoints](https://huggingface.co/models?other=mms)
- MMS base checkpoints:
  - [facebook/mms-1b](https://huggingface.co/facebook/mms-1b)
  - [facebook/mms-300m](https://huggingface.co/facebook/mms-300m)
- [Official Space](https://huggingface.co/spaces/facebook/MMS)
