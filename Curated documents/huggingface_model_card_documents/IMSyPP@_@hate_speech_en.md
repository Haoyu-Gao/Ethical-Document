---
language:
- en
license: mit
widget:
- text: My name is Mark and I live in London. I am a postgraduate student at Queen
    Mary University.
---

# Hate Speech Classifier for Social Media Content in English Language

A monolingual model for hate speech classification of social media content in English language. The model was trained on 103190 YouTube comments and tested on an independent test set of 20554 YouTube comments. It is based on English BERT base pre-trained language model.

## Please cite:
Kralj Novak, P., Scantamburlo, T., Pelicon, A., Cinelli, M., Mozetič, I., & Zollo, F. (2022, July). __Handling disagreement in hate speech modelling__. In International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems (pp. 681-695). Cham: Springer International Publishing.
https://link.springer.com/chapter/10.1007/978-3-031-08974-9_54

## Tokenizer

During training the text was preprocessed using the original English BERT base tokenizer. We suggest the same tokenizer is used for inference.

## Model output

The model classifies each input into one of four distinct classes:
* 0 - acceptable
* 1 - inappropriate
* 2 - offensive
* 3 - violent


Details on data acquisition and labeling including the Annotation guidelines:  
http://imsypp.ijs.si/wp-content/uploads/2021/12/IMSyPP_D2.2_multilingual-dataset.pdf

