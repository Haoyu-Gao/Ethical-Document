---
{}
---
# BROS

GitHub: https://github.com/clovaai/bros

## Introduction

BROS (BERT Relying On Spatiality) is a pre-trained language model focusing on text and layout for better key information extraction from documents.<br>
Given the OCR results of the document image, which are text and bounding box pairs, it can perform various key information extraction tasks, such as extracting an ordered item list from receipts.<br>
For more details, please refer to our paper:

BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents<br>
Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park<br>
AAAI 2022 - Main Technical Track

[[arXiv]](https://arxiv.org/abs/2108.04539)

## Pre-trained models
| name               | # params | Hugging Face - Models                                                                           |
|---------------------|---------:|-------------------------------------------------------------------------------------------------|
| bros-base-uncased (**this**)  |   < 110M | [naver-clova-ocr/bros-base-uncased](https://huggingface.co/naver-clova-ocr/bros-base-uncased)   |
| bros-large-uncased |   < 340M | [naver-clova-ocr/bros-large-uncased](https://huggingface.co/naver-clova-ocr/bros-large-uncased) |