---
language:
- ru
tags:
- toxic comments classification
licenses:
- cc-by-nc-sa
---

## General concept of the model

This model is trained on the dataset of sensitive topics of the Russian language. The concept of sensitive topics is described [in this article ](https://www.aclweb.org/anthology/2021.bsnlp-1.4/) presented at the workshop for Balto-Slavic NLP at the EACL-2021 conference. Please note that this article describes the first version of the dataset, while the model is trained on the extended version of the dataset open-sourced on our [GitHub](https://github.com/skoltech-nlp/inappropriate-sensitive-topics/blob/main/Version2/sensitive_topics/sensitive_topics.csv) or on [kaggle](https://www.kaggle.com/nigula/russian-sensitive-topics). The properties of the dataset is the same as the one described in the article, the only difference is the size.


## Instructions

The model predicts combinations of 18 sensitive topics described in the [article](https://arxiv.org/abs/2103.05345). You can find step-by-step instructions for using the model [here](https://github.com/skoltech-nlp/inappropriate-sensitive-topics/blob/main/Version2/sensitive_topics/Inference.ipynb)


## Metrics

The dataset partially manually labeled samples and partially semi-automatically labeled samples. Learn more in our article. We tested the performance of the classifier only on the part of manually labeled data that is why some topics are not well represented in the test set.


|                   | precision | recall | f1-score | support |
|-------------------|-----------|--------|----------|---------|
| offline_crime     |      0.65 |   0.55 |      0.6 |     132 |
| online_crime      |       0.5 |   0.46 |     0.48 |      37 |
| drugs             |      0.87 |    0.9 |     0.88 |      87 |
| gambling          |       0.5 |   0.67 |     0.57 |       6 |
| pornography       |      0.73 |   0.59 |     0.65 |     204 |
| prostitution      |      0.75 |   0.69 |     0.72 |      91 |
| slavery           |      0.72 |   0.72 |     0.73 |      40 |
| suicide           |      0.33 |   0.29 |     0.31 |       7 |
| terrorism         |      0.68 |   0.57 |     0.62 |      47 |
| weapons           |      0.89 |   0.83 |     0.86 |     138 |
| body_shaming      |       0.9 |   0.67 |     0.77 |     109 |
| health_shaming    |      0.84 |   0.55 |     0.66 |     108 |
| politics          |      0.68 |   0.54 |      0.6 |     241 |
| racism            |      0.81 |   0.59 |     0.68 |     204 |
| religion          |      0.94 |   0.72 |     0.81 |     102 |
| sexual_minorities |      0.69 |   0.46 |     0.55 |     102 |
| sexism            |      0.66 |   0.64 |     0.65 |     132 |
| social_injustice  |      0.56 |   0.37 |     0.45 |     181 |
| none              |      0.62 |   0.67 |     0.64 |     250 |
| micro avg         |      0.72 |   0.61 |     0.66 |    2218 |
| macro avg         |       0.7 |    0.6 |     0.64 |    2218 |
| weighted avg      |      0.73 |   0.61 |     0.66 |    2218 |
| samples avg       |      0.75 |   0.66 |     0.68 |    2218 |

## Licensing Information

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png

## Citation

If you find this repository helpful, feel free to cite our publication:

```
@inproceedings{babakov-etal-2021-detecting,
    title = "Detecting Inappropriate Messages on Sensitive Topics that Could Harm a Company{'}s Reputation",
    author = "Babakov, Nikolay  and
      Logacheva, Varvara  and
      Kozlova, Olga  and
      Semenov, Nikita  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.4",
    pages = "26--36",
    abstract = "Not all topics are equally {``}flammable{''} in terms of toxicity: a calm discussion of turtles or fishing less often fuels inappropriate toxic dialogues than a discussion of politics or sexual minorities. We define a set of sensitive topics that can yield inappropriate and toxic messages and describe the methodology of collecting and labelling a dataset for appropriateness. While toxicity in user-generated data is well-studied, we aim at defining a more fine-grained notion of inappropriateness. The core of inappropriateness is that it can harm the reputation of a speaker. This is different from toxicity in two respects: (i) inappropriateness is topic-related, and (ii) inappropriate message is not toxic but still unacceptable. We collect and release two datasets for Russian: a topic-labelled dataset and an appropriateness-labelled dataset. We also release pre-trained classification models trained on this data.",
}
```