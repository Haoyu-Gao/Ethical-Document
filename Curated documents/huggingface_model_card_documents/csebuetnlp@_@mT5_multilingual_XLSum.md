---
language:
- am
- ar
- az
- bn
- my
- zh
- en
- fr
- gu
- ha
- hi
- ig
- id
- ja
- rn
- ko
- ky
- mr
- ne
- om
- ps
- fa
- pcm
- pt
- pa
- ru
- gd
- sr
- si
- so
- es
- sw
- ta
- te
- th
- ti
- tr
- uk
- ur
- uz
- vi
- cy
- yo
tags:
- summarization
- mT5
datasets:
- csebuetnlp/xlsum
licenses:
- cc-by-nc-sa-4.0
widget:
- text: Videos that say approved vaccines are dangerous and cause autism, cancer or
    infertility are among those that will be taken down, the company said.  The policy
    includes the termination of accounts of anti-vaccine influencers.  Tech giants
    have been criticised for not doing more to counter false health information on
    their sites.  In July, US President Joe Biden said social media platforms were
    largely responsible for people's scepticism in getting vaccinated by spreading
    misinformation, and appealed for them to address the issue.  YouTube, which is
    owned by Google, said 130,000 videos were removed from its platform since last
    year, when it implemented a ban on content spreading misinformation about Covid
    vaccines.  In a blog post, the company said it had seen false claims about Covid
    jabs "spill over into misinformation about vaccines in general". The new policy
    covers long-approved vaccines, such as those against measles or hepatitis B.  "We're
    expanding our medical misinformation policies on YouTube with new guidelines on
    currently administered vaccines that are approved and confirmed to be safe and
    effective by local health authorities and the WHO," the post said, referring to
    the World Health Organization.
model-index:
- name: csebuetnlp/mT5_multilingual_XLSum
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: xsum
      type: xsum
      config: default
      split: test
    metrics:
    - type: rouge
      value: 36.5002
      name: ROUGE-1
      verified: true
    - type: rouge
      value: 13.934
      name: ROUGE-2
      verified: true
    - type: rouge
      value: 28.9876
      name: ROUGE-L
      verified: true
    - type: rouge
      value: 28.9958
      name: ROUGE-LSUM
      verified: true
    - type: loss
      value: 2.0674800872802734
      name: loss
      verified: true
    - type: gen_len
      value: 26.9733
      name: gen_len
      verified: true
---

# mT5-multilingual-XLSum

This repository contains the mT5 checkpoint finetuned on the 45 languages of [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) dataset. For finetuning details and scripts,
see the [paper](https://aclanthology.org/2021.findings-acl.413/) and the [official repository](https://github.com/csebuetnlp/xl-sum). 


## Using this model in `transformers` (tested on 4.11.0.dev0)

```python
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)
```

## Benchmarks

Scores on the XL-Sum test sets are as follows:

Language | ROUGE-1 / ROUGE-2 / ROUGE-L
---------|----------------------------
Amharic | 20.0485 / 7.4111 / 18.0753
Arabic | 34.9107 / 14.7937 / 29.1623
Azerbaijani | 21.4227 / 9.5214 / 19.3331
Bengali | 29.5653 / 12.1095 / 25.1315
Burmese | 15.9626 / 5.1477 / 14.1819
Chinese (Simplified) | 39.4071 / 17.7913 / 33.406
Chinese (Traditional) | 37.1866 / 17.1432 / 31.6184
English | 37.601 / 15.1536 / 29.8817
French | 35.3398 / 16.1739 / 28.2041
Gujarati | 21.9619 / 7.7417 / 19.86
Hausa | 39.4375 / 17.6786 / 31.6667
Hindi | 38.5882 / 16.8802 / 32.0132
Igbo | 31.6148 / 10.1605 / 24.5309
Indonesian | 37.0049 / 17.0181 / 30.7561
Japanese | 48.1544 / 23.8482 / 37.3636
Kirundi | 31.9907 / 14.3685 / 25.8305
Korean | 23.6745 / 11.4478 / 22.3619
Kyrgyz | 18.3751 / 7.9608 / 16.5033
Marathi | 22.0141 / 9.5439 / 19.9208
Nepali | 26.6547 / 10.2479 / 24.2847
Oromo | 18.7025 / 6.1694 / 16.1862
Pashto | 38.4743 / 15.5475 / 31.9065
Persian | 36.9425 / 16.1934 / 30.0701
Pidgin | 37.9574 / 15.1234 / 29.872
Portuguese | 37.1676 / 15.9022 / 28.5586
Punjabi | 30.6973 / 12.2058 / 25.515
Russian | 32.2164 / 13.6386 / 26.1689
Scottish Gaelic | 29.0231 / 10.9893 / 22.8814
Serbian (Cyrillic) | 23.7841 / 7.9816 / 20.1379
Serbian (Latin) | 21.6443 / 6.6573 / 18.2336
Sinhala | 27.2901 / 13.3815 / 23.4699
Somali | 31.5563 / 11.5818 / 24.2232
Spanish | 31.5071 / 11.8767 / 24.0746
Swahili | 37.6673 / 17.8534 / 30.9146
Tamil | 24.3326 / 11.0553 / 22.0741
Telugu | 19.8571 / 7.0337 / 17.6101
Thai | 37.3951 / 17.275 / 28.8796
Tigrinya | 25.321 / 8.0157 / 21.1729
Turkish | 32.9304 / 15.5709 / 29.2622
Ukrainian | 23.9908 / 10.1431 / 20.9199
Urdu | 39.5579 / 18.3733 / 32.8442
Uzbek | 16.8281 / 6.3406 / 15.4055
Vietnamese | 32.8826 / 16.2247 / 26.0844
Welsh | 32.6599 / 11.596 / 26.1164
Yoruba | 31.6595 / 11.6599 / 25.0898



## Citation

If you use this model, please cite the following paper:
```
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
```