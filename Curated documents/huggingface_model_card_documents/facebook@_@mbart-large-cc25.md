---
language:
- en
- ar
- cs
- de
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
- multilingual
tags:
- translation
---
#### mbart-large-cc25

Pretrained (not finetuned) multilingual mbart model.
Original Languages
```
export langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
```

Original Code: https://github.com/pytorch/fairseq/tree/master/examples/mbart
Docs:  https://huggingface.co/transformers/master/model_doc/mbart.html
Finetuning Code: examples/seq2seq/finetune.py (as of Aug 20, 2020)

Can also be finetuned for summarization.