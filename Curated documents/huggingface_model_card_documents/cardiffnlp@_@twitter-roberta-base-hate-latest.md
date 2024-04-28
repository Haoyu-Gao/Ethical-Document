---
language:
- en
pipeline_tag: text-classification
model-index:
- name: twitter-roberta-base-hate-latest
  results: []
---
# cardiffnlp/twitter-roberta-base-hate-latest

This model is a fine-tuned version of [cardiffnlp/twitter-roberta-base-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2022-154m) for binary hate-speech classification. 
A combination of 13 different hate-speech datasets in the English language were used to fine-tune the model. 
More details in the [reference paper](https://aclanthology.org/2023.woah-1.25/).

| **Dataset**   |   **Accuracy** |   **Macro-F1** | **Weighted-F1** |
|:----------|-----------:|-----------:|--------------:|
| hatEval, SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter      |     0.5831 |     0.5646 |        0.548  |
| ucberkeley-dlab/measuring-hate-speech         |     0.9273 |     0.9193 |        0.928  |
| Detecting East Asian Prejudice on Social Media       |     0.9231 |     0.6623 |        0.9428 |
| Call me sexist, but         |     0.9686 |     0.9203 |        0.9696 |
| Predicting the Type and Target of Offensive Posts in Social Media   |     0.9164 |     0.6847 |        0.9098 |
| HateXplain     |     0.8653 |     0.845  |        0.8662 |
| Large Scale Crowdsourcing and Characterization of Twitter Abusive BehaviorLarge Scale Crowdsourcing and Characterization of Twitter Abusive Behavior       |     0.7801 |     0.7446 |        0.7614 |
| Multilingual and Multi-Aspect Hate Speech Analysis      |     0.9944 |     0.4986 |        0.9972 |
| Hate speech and offensive content identification in indo-european languages     |     0.8779 |     0.6904 |        0.8706 |
| Are You a Racist or Am I Seeing Things?       |     0.921  |     0.8935 |        0.9216 |
| Automated Hate Speech Detection      |     0.9423 |     0.9249 |        0.9429 |
| Hate Towards the Political Opponent      |     0.8783 |     0.6595 |        0.8788 |
| Hateful Symbols or Hateful People?      |     0.8187 |     0.7833 |        0.8323 |
| **Overall**                                                                                                                                          |  **0.8766**  |  **0.7531**  |    **0.8745**   |



### Usage
Install tweetnlp via pip.
```shell
pip install tweetnlp
```
Load the model in python.
```python
import tweetnlp
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)')
>> {'label': 'NOT-HATE'}

```


### Reference paper - Model based on:
```
@inproceedings{antypas-camacho-collados-2023-robust,
    title = "Robust Hate Speech Detection in Social Media: A Cross-Dataset Empirical Evaluation",
    author = "Antypas, Dimosthenis  and
      Camacho-Collados, Jose",
    booktitle = "The 7th Workshop on Online Abuse and Harms (WOAH)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.woah-1.25",
    pages = "231--242"
}

```