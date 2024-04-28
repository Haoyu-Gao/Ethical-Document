---
language:
- ja
license: apache-2.0
library_name: transformers
datasets:
- llm-book/aio-retriever
pipeline_tag: feature-extraction
---

# bert-base-japanese-v3-unsup-simcse-jawiki

「[大規模言語モデル入門](https://www.amazon.co.jp/dp/4297136333)」の第8章で紹介している教師なしSimCSEのモデルです。
[cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3) を [llm-book/jawiki-sentences](https://huggingface.co/datasets/llm-book/jawiki-sentences) でファインチューニングして構築されています。


## 関連リンク

* [GitHubリポジトリ](https://github.com/ghmagazine/llm-book)
* [Colabノートブック（訓練）](https://colab.research.google.com/github/ghmagazine/llm-book/blob/main/chapter8/8-3-simcse-training.ipynb)
* [Colabノートブック（推論）](https://colab.research.google.com/github/ghmagazine/llm-book/blob/main/chapter8/8-4-simcse-faiss.ipynb)
* [データセット](https://huggingface.co/datasets/llm-book/jawiki-sentences)
* [大規模言語モデル入門（Amazon.co.jp）](https://www.amazon.co.jp/dp/4297136333/)
* [大規模言語モデル入門（gihyo.jp）](https://gihyo.jp/book/2023/978-4-297-13633-8)


## 使い方

```py
from torch.nn.functional import cosine_similarity
from transformers import pipeline

sim_enc_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-unsup-simcse-jawiki", task="feature-extraction")

text = "川べりでサーフボードを持った人たちがいます"
sim_text = "サーファーたちが川べりに立っています"

# text と sim_text のベクトルを獲得
text_emb = sim_enc_pipeline(text, return_tensors=True)[0][0]
sim_emb = sim_enc_pipeline(sim_text, return_tensors=True)[0][0]
# text と sim_text の類似度を計算
sim_pair_score = cosine_similarity(text_emb, sim_emb, dim=0)
print(sim_pair_score.item())  # -> 0.8568589687347412
```


## ライセンス

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
