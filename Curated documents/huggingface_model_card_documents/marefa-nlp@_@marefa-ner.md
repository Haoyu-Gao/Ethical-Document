---
language: ar
datasets:
- Marefa-NER
widget:
- text: في استاد القاهرة، بدأ حفل افتتاح بطولة كأس الأمم الأفريقية بحضور رئيس الجمهورية
    و رئيس الاتحاد الدولي لكرة القدم
---

# Tebyan تبيـان
## Marefa Arabic Named Entity Recognition Model
## نموذج المعرفة لتصنيف أجزاء النص

<p align="center">
<img src="https://huggingface.co/marefa-nlp/marefa-ner/resolve/main/assets/marefa-tebyan-banner.png" alt="Marfa Arabic NER Model" width="600"/>
</p?
 
---------
**Version**: 1.3

**Last Update:** 3-12-2021

## Model description

**Marefa-NER** is a Large Arabic Named Entity Recognition (NER) model built on a completely new dataset and targets to extract up to 9 different types of entities
```
Person, Location, Organization, Nationality, Job, Product, Event, Time, Art-Work
```

نموذج المعرفة لتصنيف أجزاء النص. نموذج جديد كليا من حيث البيانات المستخدمة في تدريب النموذج. 
كذلك يستهدف النموذج تصنيف حتى 9 أنواع مختلفة من أجزاء النص
```
شخص - مكان - منظمة - جنسية - وظيفة - منتج - حدث - توقيت - عمل إبداعي
```

## How to use كيف تستخدم النموذج

*You can test the model quickly by checking this [Colab notebook](https://colab.research.google.com/drive/1OGp9Wgm-oBM5BBhTLx6Qow4dNRSJZ-F5?usp=sharing)*

----

Install the following Python packages

`$ pip3 install transformers==4.8.0 nltk==3.5 protobuf==3.15.3 torch==1.9.0 `

> If you are using `Google Colab`, please restart your runtime after installing the packages.


-----------

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

custom_labels = ["O", "B-job", "I-job", "B-nationality", "B-person", "I-person", "B-location","B-time", "I-time", "B-event", "I-event", "B-organization", "I-organization", "I-location", "I-nationality", "B-product", "I-product", "B-artwork", "I-artwork"]

def _extract_ner(text: str, model: AutoModelForTokenClassification,
                 tokenizer: AutoTokenizer, start_token: str="▁"):
    tokenized_sentence = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    tokenized_sentences = tokenized_sentence['input_ids'].numpy()

    with torch.no_grad():
        output = model(**tokenized_sentence)

    last_hidden_states = output[0].numpy()
    label_indices = np.argmax(last_hidden_states[0], axis=1)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentences[0])
    special_tags = set(tokenizer.special_tokens_map.values())

    grouped_tokens = []
    for token, label_idx in zip(tokens, label_indices):
        if token not in special_tags:
            if not token.startswith(start_token) and len(token.replace(start_token,"").strip()) > 0:
                grouped_tokens[-1]["token"] += token
            else:
                grouped_tokens.append({"token": token, "label": custom_labels[label_idx]})

    # extract entities
    ents = []
    prev_label = "O"
    for token in grouped_tokens:
        label = token["label"].replace("I-","").replace("B-","")
        if token["label"] != "O":
            
            if label != prev_label:
                ents.append({"token": [token["token"]], "label": label})
            else:
                ents[-1]["token"].append(token["token"])
            
        prev_label = label
    
    # group tokens
    ents = [{"token": "".join(rec["token"]).replace(start_token," ").strip(), "label": rec["label"]}  for rec in ents ]

    return ents

model_cp = "marefa-nlp/marefa-ner"

tokenizer = AutoTokenizer.from_pretrained(model_cp)
model = AutoModelForTokenClassification.from_pretrained(model_cp, num_labels=len(custom_labels))

samples = [
    "تلقى تعليمه في الكتاب ثم انضم الى الأزهر عام 1873م. تعلم على يد السيد جمال الدين الأفغاني والشيخ محمد عبده",
    "بعد عودته إلى القاهرة، التحق نجيب الريحاني فرقة جورج أبيض، الذي كان قد ضمَّ - قُبيل ذلك - فرقته إلى فرقة سلامة حجازي . و منها ذاع صيته",
    "في استاد القاهرة، قام حفل افتتاح بطولة كأس الأمم الأفريقية بحضور رئيس الجمهورية و رئيس الاتحاد الدولي لكرة القدم",
    "من فضلك أرسل هذا البريد الى صديقي جلال الدين في تمام الساعة الخامسة صباحا في يوم الثلاثاء القادم",
    "امبارح اتفرجت على مباراة مانشستر يونايتد مع ريال مدريد في غياب الدون كرستيانو رونالدو",
    "لا تنسى تصحيني الساعة سبعة, و ضيف في الجدول اني احضر مباراة نادي النصر غدا",
]

# [optional]
samples = [ " ".join(word_tokenize(sample.strip())) for sample in samples if sample.strip() != "" ]

for sample in samples:
    ents = _extract_ner(text=sample, model=model, tokenizer=tokenizer, start_token="▁")

    print(sample)
    for ent in ents:
        print("\t",ent["token"],"==>",ent["label"])
    print("========\n")

```

Output

```
تلقى تعليمه في الكتاب ثم انضم الى الأزهر عام 1873م . تعلم على يد السيد جمال الدين الأفغاني والشيخ محمد عبده
	 الأزهر ==> organization
	 عام 1873م ==> time
	 السيد جمال الدين الأفغاني ==> person
	 محمد عبده ==> person
========

بعد عودته إلى القاهرة، التحق نجيب الريحاني فرقة جورج أبيض، الذي كان قد ضمَّ - قُبيل ذلك - فرقته إلى فرقة سلامة حجازي . و منها ذاع صيته
	 القاهرة، ==> location
	 نجيب الريحاني ==> person
	 فرقة جورج أبيض، ==> organization
	 فرقة سلامة حجازي ==> organization
========

في استاد القاهرة، قام حفل افتتاح بطولة كأس الأمم الأفريقية بحضور رئيس الجمهورية و رئيس الاتحاد الدولي لكرة القدم
	 استاد القاهرة، ==> location
	 بطولة كأس الأمم الأفريقية ==> event
	 رئيس الجمهورية ==> job
	 رئيس ==> job
	 الاتحاد الدولي لكرة القدم ==> organization
========

من فضلك أرسل هذا البريد الى صديقي جلال الدين في تمام الساعة الخامسة صباحا في يوم الثلاثاء القادم
	 جلال الدين ==> person
	 الساعة الخامسة صباحا ==> time
	 يوم الثلاثاء القادم ==> time
========

امبارح اتفرجت على مباراة مانشستر يونايتد مع ريال مدريد في غياب الدون كرستيانو رونالدو
	 مانشستر يونايتد ==> organization
	 ريال مدريد ==> organization
	 كرستيانو رونالدو ==> person
========

لا تنسى تصحيني الساعة سبعة , و ضيف في الجدول اني احضر مباراة نادي النصر غدا
	 الساعة سبعة ==> time
	 نادي النصر ==> organization
	 غدا ==> time
========
```

## Fine-Tuning

Check this [notebook](https://colab.research.google.com/drive/1WUYrnmDFFEItqGMvbyjqZEJJqwU7xQR-?usp=sharing) to fine-tune the NER model

## Evaluation

We tested the model agains a test set of 1959 sentences. The results is in the follwing table

| type         |   f1-score |   precision |   recall |   support |
|:-------------|-----------:|------------:|---------:|----------:|
| person       |   0.93298  |    0.931479 | 0.934487 |      4335 |
| location     |   0.891537 |    0.896926 | 0.886212 |      4939 |
| time         |   0.873003 |    0.876087 | 0.869941 |      1853 |
| nationality  |   0.871246 |    0.843153 | 0.901277 |      2350 |
| job          |   0.837656 |    0.79912  | 0.880097 |      2477 |
| organization |   0.781317 |    0.773328 | 0.789474 |      2299 |
| event        |   0.686695 |    0.733945 | 0.645161 |       744 |
| artwork      |   0.653552 |    0.678005 | 0.630802 |       474 |
| product      |   0.625483 |    0.553531 | 0.718935 |       338 |
| **weighted avg** |   0.859008 |    0.852365 | 0.86703  |     19809 |
| **micro avg**    |   0.858771 |    0.850669 | 0.86703  |     19809 |
| **macro avg**   |   0.79483  |    0.787286 | 0.806265 |     19809 |

## Acknowledgment شكر و تقدير

قام بإعداد البيانات التي تم تدريب النموذج عليها, مجموعة من المتطوعين الذين قضوا ساعات يقومون بتنقيح البيانات و مراجعتها

- على سيد عبد الحفيظ - إشراف
- نرمين محمد عطيه 
- صلاح خيرالله
- احمد علي عبدربه
- عمر بن عبد العزيز سليمان
- محمد ابراهيم الجمال
- عبدالرحمن سلامه خلف
- إبراهيم كمال محمد سليمان
- حسن مصطفى حسن 
- أحمد فتحي سيد
- عثمان مندو
- عارف الشريف
- أميرة محمد محمود
- حسن سعيد حسن
- عبد العزيز علي البغدادي
- واثق عبدالملك الشويطر
- عمرو رمضان عقل الحفناوي
- حسام الدين أحمد على
- أسامه أحمد محمد محمد
- حاتم محمد المفتي
- عبد الله دردير
- أدهم البغدادي
- أحمد صبري
- عبدالوهاب محمد محمد
- أحمد محمد عوض