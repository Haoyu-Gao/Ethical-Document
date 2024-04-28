---
language:
- ja
license: apache-2.0
library_name: transformers
datasets:
- llm-book/ner-wikipedia-dataset
metrics:
- seqeval
- precision
- recall
- f1
pipeline_tag: token-classification
---

# llm-book/bert-base-japanese-v3-ner-wikipedia-dataset

「[大規模言語モデル入門](https://www.amazon.co.jp/dp/4297136333)」の第6章で紹介している固有表現認識のモデルです。
[cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)を[llm-book/ner-wikipedia-dataset](https://huggingface.co/datasets/llm-book/ner-wikipedia-dataset)でファインチューニングして構築されています。

## 関連リンク

* [GitHubリポジトリ](https://github.com/ghmagazine/llm-book)
* [Colabノートブック](https://colab.research.google.com/github/ghmagazine/llm-book/blob/main/chapter6/6-named-entity-recognition.ipynb)
* [データセット](https://huggingface.co/datasets/llm-book/ner-wikipedia-dataset)
* [大規模言語モデル入門（Amazon.co.jp）](https://www.amazon.co.jp/dp/4297136333/)
* [大規模言語モデル入門（gihyo.jp）](https://gihyo.jp/book/2023/978-4-297-13633-8)

## 使い方
```python
from transformers import pipeline
from pprint import pprint

ner_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)
text = "大谷翔平は岩手県水沢市出身のプロ野球選手"
# text中の固有表現を抽出
pprint(ner_pipeline(text))
# [{'end': None,
#   'entity_group': '人名',
#   'score': 0.99823624,
#   'start': None,
#   'word': '大谷 翔平'},
#  {'end': None,
#   'entity_group': '地名',
#   'score': 0.9986874,
#   'start': None,
#   'word': '岩手 県 水沢 市'}]
```

## ライセンス

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)