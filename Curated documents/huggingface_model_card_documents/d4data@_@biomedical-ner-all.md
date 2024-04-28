---
language:
- en
license: apache-2.0
tags:
- Token Classification
co2_eq_emissions: 0.0279399890043426
widget:
- text: 'CASE: A 28-year-old previously healthy man presented with a 6-week history
    of palpitations. The symptoms occurred during rest, 2–3 times per week, lasted
    up to 30 minutes at a time and were associated with dyspnea. Except for a grade
    2/6 holosystolic tricuspid regurgitation murmur (best heard at the left sternal
    border with inspiratory accentuation), physical examination yielded unremarkable
    findings.'
  example_title: example 1
- text: A 63-year-old woman with no known cardiac history presented with a sudden
    onset of dyspnea requiring intubation and ventilatory support out of hospital.
    She denied preceding symptoms of chest discomfort, palpitations, syncope or infection.
    The patient was afebrile and normotensive, with a sinus tachycardia of 140 beats/min.
  example_title: example 2
- text: A 48 year-old female presented with vaginal bleeding and abnormal Pap smears.
    Upon diagnosis of invasive non-keratinizing SCC of the cervix, she underwent a
    radical hysterectomy with salpingo-oophorectomy which demonstrated positive spread
    to the pelvic lymph nodes and the parametrium. Pathological examination revealed
    that the tumour also extensively involved the lower uterine segment.
  example_title: example 3
---

## About the Model
An English Named Entity Recognition model, trained on Maccrobat to recognize the bio-medical entities (107 entities) from a given text corpus (case reports etc.). This model was built on top of distilbert-base-uncased

- Dataset: Maccrobat https://figshare.com/articles/dataset/MACCROBAT2018/9764942
- Carbon emission: 0.0279399890043426 Kg
- Training time: 30.16527 minutes
- GPU used : 1 x GeForce RTX 3060 Laptop GPU

Checkout the tutorial video for explanation of this model and corresponding python library: https://youtu.be/xpiDPdBpS18

## Usage
The easiest way is to load the inference api from huggingface and second method is through the pipeline object offered by transformers library.
```python
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu
pipe("""The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.""")
```

## Author
This model is part of the Research topic "AI in Biomedical field" conducted by Deepak John Reji, Shaina Raza. If you use this work (code, model or dataset), please star at:
> https://github.com/dreji18/Bio-Epidemiology-NER

## You can support me here :)
<a href="https://www.buymeacoffee.com/deepakjohnreji" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>