---
language: en
tags:
- bert
- medical
- clinical
- assertion
- negation
- text-classification
widget:
- text: Patient denies [entity] SOB [entity].
---

# Clinical Assertion / Negation Classification BERT

## Model description

The Clinical Assertion and Negation Classification BERT is introduced in the paper [Assertion Detection in Clinical Notes: Medical Language Models to the Rescue?
](https://aclanthology.org/2021.nlpmc-1.5/). The model helps structure information in clinical patient letters by classifying medical conditions mentioned in the letter into PRESENT, ABSENT and POSSIBLE.

The model is based on the [ClinicalBERT - Bio + Discharge Summary BERT Model](https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT) by Alsentzer et al. and fine-tuned on assertion data from the [2010 i2b2 challenge](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3168320/).


#### How to use the model

You can load the model via the transformers library:
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")

```

The model expects input in the form of spans/sentences with one marked entity to classify as `PRESENT(0)`, `ABSENT(1)` or `POSSIBLE(2)`. The entity in question is identified with the special token `[entity]` surrounding it.

Example input and inference:
```
input = "The patient recovered during the night and now denies any [entity] shortness of breath [entity]."

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

classification = classifier(input)
# [{'label': 'ABSENT', 'score': 0.9842607378959656}]
``` 

### Cite

When working with the model, please cite our paper as follows:

```bibtex
@inproceedings{van-aken-2021-assertion,
    title = "Assertion Detection in Clinical Notes: Medical Language Models to the Rescue?",
    author = "van Aken, Betty  and
      Trajanovska, Ivana  and
      Siu, Amy  and
      Mayrdorfer, Manuel  and
      Budde, Klemens  and
      Loeser, Alexander",
    booktitle = "Proceedings of the Second Workshop on Natural Language Processing for Medical Conversations",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nlpmc-1.5",
    doi = "10.18653/v1/2021.nlpmc-1.5"
}
```