---
language:
- multilingual
- ar
- cs
- de
- en
- es
- et
- fi
- fr
- gu
- hi
- it
- ja
- kk
- ko
- lt
- lv
- my
- ne
- nl
- ro
- ru
- si
- tr
- vi
- zh
- af
- az
- bn
- fa
- he
- hr
- id
- ka
- km
- mk
- ml
- mn
- mr
- pl
- ps
- pt
- sv
- sw
- ta
- te
- th
- tl
- uk
- ur
- xh
- gl
- sl
license: mit
tags:
- mbart-50
---

# mBART-50

mBART-50 is a multilingual Sequence-to-Sequence model pre-trained using the "Multilingual Denoising Pretraining" objective. It was introduced in [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401) paper.

## Model description

mBART-50 is a multilingual Sequence-to-Sequence model. It was introduced to show that multilingual translation models can be created through multilingual fine-tuning. 
Instead of fine-tuning on one direction, a pre-trained model is fine-tuned on many directions simultaneously. mBART-50 is created using the original mBART model and extended to add extra 25 languages to support multilingual machine translation models of 50 languages. The pre-training objective is explained below.

**Multilingual Denoising Pretraining**: The model incorporates N languages by concatenating data: 
`D = {D1, ..., DN }` where each Di is a collection of monolingual documents in language `i`. The source documents are noised using two schemes, 
first randomly shuffling the original sentences' order, and second a novel in-filling scheme, 
where spans of text are replaced with a single mask token. The model is then tasked to reconstruct the original text. 
35% of each instance's words are masked by random sampling a span length according to a Poisson distribution `(λ = 3.5)`.
The decoder input is the original text with one position offset. A language id symbol `LID` is used as the initial token to predict the sentence.


## Intended uses & limitations

`mbart-large-50` is pre-trained model and primarily aimed at being fine-tuned on translation tasks. It can also be fine-tuned on other multilingual sequence-to-sequence tasks. 
See the [model hub](https://huggingface.co/models?filter=mbart-50) to look for fine-tuned versions.


## Training

As the model is multilingual, it expects the sequences in a different format. A special language id token is used as a prefix in both the source and target text. The text format is `[lang_code] X [eos]` with `X` being the source or target text respectively and `lang_code` is `source_lang_code` for source text and `tgt_lang_code` for target text. `bos` is never used. Once the examples are prepared in this format, it can be trained as any other sequence-to-sequence model.


```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_text, return_tensors="pt").input_ids

model(**model_inputs, labels=labels) # forward pass
```



## Languages covered
Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)


## BibTeX entry and citation info
```
@article{tang2020multilingual,
    title={Multilingual Translation with Extensible Multilingual Pretraining and Finetuning},
    author={Yuqing Tang and Chau Tran and Xian Li and Peng-Jen Chen and Naman Goyal and Vishrav Chaudhary and Jiatao Gu and Angela Fan},
    year={2020},
    eprint={2008.00401},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```