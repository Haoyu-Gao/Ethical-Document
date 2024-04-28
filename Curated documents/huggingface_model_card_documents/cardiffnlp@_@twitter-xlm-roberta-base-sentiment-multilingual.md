---
datasets:
- cardiffnlp/tweet_sentiment_multilingual
metrics:
- f1
- accuracy
pipeline_tag: text-classification
widget:
- text: Get the all-analog Classic Vinyl Edition of "Takin Off" Album from {@herbiehancock@}
    via {@bluenoterecords@} link below {{URL}}
  example_title: topic_classification 1
- text: Yes, including Medicare and social security saving👍
  example_title: sentiment 1
- text: All two of them taste like ass.
  example_title: offensive 1
- text: If you wanna look like a badass, have drama on social media
  example_title: irony 1
- text: Whoever just unfollowed me you a bitch
  example_title: hate 1
- text: I love swimming for the same reason I love meditating...the feeling of weightlessness.
  example_title: emotion 1
- text: Beautiful sunset last night from the pontoon @TupperLakeNY
  example_title: emoji 1
model-index:
- name: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: cardiffnlp/tweet_sentiment_multilingual
      type: all
      split: test
    metrics:
    - type: micro_f1_cardiffnlp/tweet_sentiment_multilingual/all
      value: 0.6931034482758621
      name: Micro F1 (cardiffnlp/tweet_sentiment_multilingual/all)
    - type: micro_f1_cardiffnlp/tweet_sentiment_multilingual/all
      value: 0.692628774202147
      name: Macro F1 (cardiffnlp/tweet_sentiment_multilingual/all)
    - type: accuracy_cardiffnlp/tweet_sentiment_multilingual/all
      value: 0.6931034482758621
      name: Accuracy (cardiffnlp/tweet_sentiment_multilingual/all)
---
# cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual 

This model is a fine-tuned version of [cardiffnlp/twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base) on the 
[`cardiffnlp/tweet_sentiment_multilingual (all)`](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) 
via [`tweetnlp`](https://github.com/cardiffnlp/tweetnlp).
Training split is `train` and parameters have been tuned on the validation split `validation`.

Following metrics are achieved on the test split `test` ([link](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual/raw/main/metric.json)).

- F1 (micro): 0.6931034482758621
- F1 (macro): 0.692628774202147
- Accuracy: 0.6931034482758621

### Usage
Install tweetnlp via pip.
```shell
pip install tweetnlp
```
Load the model in python.
```python
import tweetnlp
model = tweetnlp.Classifier("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", max_length=128)
model.predict('Get the all-analog Classic Vinyl Edition of "Takin Off" Album from {@herbiehancock@} via {@bluenoterecords@} link below {{URL}}')
```

### Reference

```
@inproceedings{dimosthenis-etal-2022-twitter,
    title = "{T}witter {T}opic {C}lassification",
    author = "Antypas, Dimosthenis  and
    Ushio, Asahi  and
    Camacho-Collados, Jose  and
    Neves, Leonardo  and
    Silva, Vitor  and
    Barbieri, Francesco",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics"
}
```


