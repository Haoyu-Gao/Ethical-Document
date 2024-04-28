---
language: en
license: apache-2.0
tags:
- exbert
- security
- cybersecurity
- cyber security
- threat hunting
- threat intelligence
datasets:
- APTnotes
- Stucco-Data
- CASIE
thumbnail: https://github.com/jackaduma
---

# SecRoBERTa

This is the pretrained model presented in [SecBERT: A Pretrained Language Model for Cyber Security Text](https://github.com/jackaduma/SecBERT/), which is a SecRoBERTa model trained on cyber security text.

The training corpus was papers taken from 
 * [APTnotes](https://github.com/kbandla/APTnotes)
 * [Stucco-Data: Cyber security data sources](https://stucco.github.io/data/)
 * [CASIE: Extracting Cybersecurity Event Information from Text](https://ebiquity.umbc.edu/_file_directory_/papers/943.pdf)
 * [SemEval-2018 Task 8: Semantic Extraction from CybersecUrity REports using Natural Language Processing (SecureNLP)](https://competitions.codalab.org/competitions/17262). 

SecRoBERTa has its own wordpiece vocabulary (secvocab) that's built to best match the training corpus. 

We trained [SecBERT](https://huggingface.co/jackaduma/SecBERT) and [SecRoBERTa](https://huggingface.co/jackaduma/SecRoBERTa) versions. 

Available models include:
* [`SecBERT`](https://huggingface.co/jackaduma/SecBERT)
* [`SecRoBERTa`](https://huggingface.co/jackaduma/SecRoBERTa)

---
## **Fill Mask**

We proposed to build language model which work on cyber security text, as result, it can improve downstream tasks (NER, Text Classification, Semantic Understand, Q&A) in Cyber Security Domain.

First, as below shows Fill-Mask pipeline in [Google Bert](), [AllenAI SciBert](https://github.com/allenai/scibert) and our [SecBERT](https://github.com/jackaduma/SecBERT) .


<!-- <img src="./fill-mask-result.png" width="150%" height="150%"> -->

![fill-mask-result](https://github.com/jackaduma/SecBERT/blob/main/fill-mask-result.png?raw=true)
---

The original repo can be found [here](https://github.com/jackaduma/SecBERT).