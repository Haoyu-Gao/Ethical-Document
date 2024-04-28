---
language: en
license: mit
datasets:
- cardiffnlp/tweet_topic_multi
metrics:
- f1
- accuracy
widget:
- text: It is great to see athletes promoting awareness for climate change.
pipeline_tag: text-classification
---

# tweet-topic-21-multi

This model is based on a [TimeLMs](https://github.com/cardiffnlp/timelms) language model trained on ~124M tweets from January 2018 to December 2021 (see [here](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m)), and finetuned for multi-label topic classification on a corpus of 11,267 [tweets](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi). This model is suitable for English. 

 - Reference Paper: [TweetTopic](https://arxiv.org/abs/2209.09824) (COLING 2022). 

<b>Labels</b>: 


| <span style="font-weight:normal">0: arts_&_culture</span>           | <span style="font-weight:normal">5: fashion_&_style</span>   | <span style="font-weight:normal">10: learning_&_educational</span>  | <span style="font-weight:normal">15: science_&_technology</span>  |
|-----------------------------|---------------------|----------------------------|--------------------------|
| 1: business_&_entrepreneurs | 6: film_tv_&_video  | 11: music                  | 16: sports               |
| 2: celebrity_&_pop_culture  | 7: fitness_&_health | 12: news_&_social_concern  | 17: travel_&_adventure   |
| 3: diaries_&_daily_life     | 8: food_&_dining    | 13: other_hobbies          | 18: youth_&_student_life |
| 4: family                   | 9: gaming           | 14: relationships          |                          |


## Full classification example

```python
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit

    
MODEL = f"cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
class_mapping = model.config.id2label

text = "It is great to see athletes promoting awareness for climate change."
tokens = tokenizer(text, return_tensors='pt')
output = model(**tokens)

scores = output[0][0].detach().numpy()
scores = expit(scores)
predictions = (scores >= 0.5) * 1


# TF
#tf_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
#class_mapping = tf_model.config.id2label
#text = "It is great to see athletes promoting awareness for climate change."
#tokens = tokenizer(text, return_tensors='tf')
#output = tf_model(**tokens)
#scores = output[0][0]
#scores = expit(scores)
#predictions = (scores >= 0.5) * 1

# Map to classes
for i in range(len(predictions)):
  if predictions[i]:
    print(class_mapping[i])

```
Output: 

```
news_&_social_concern
sports
```

### BibTeX entry and citation info

Please cite the [reference paper](https://aclanthology.org/2022.coling-1.299/) if you use this model.

```bibtex
@inproceedings{antypas-etal-2022-twitter,
    title = "{T}witter Topic Classification",
    author = "Antypas, Dimosthenis  and
      Ushio, Asahi  and
      Camacho-Collados, Jose  and
      Silva, Vitor  and
      Neves, Leonardo  and
      Barbieri, Francesco",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.299",
    pages = "3386--3400"
}
```