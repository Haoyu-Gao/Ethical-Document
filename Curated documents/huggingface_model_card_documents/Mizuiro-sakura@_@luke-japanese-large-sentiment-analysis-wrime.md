---
language: ja
license: mit
tags:
- luke
- sentiment-analysis
- wrime
- SentimentAnalysis
- pytorch
- sentiment-classification
datasets: shunk031/wrime
---

# このモデルはLuke-japanese-large-liteをファインチューニングしたものです。
このモデルは８つの感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）の内、どの感情が文章に含まれているのか分析することができます。
このモデルはwrimeデータセット（
https://huggingface.co/datasets/shunk031/wrime
）を用いて学習を行いました。

# This model is based on Luke-japanese-large-lite
This model is fine-tuned model which besed on studio-ousia/Luke-japanese-large-lite.
This could be able to analyze which emotions (joy or sadness or anticipation or surprise or anger or fear or disdust or trust ) are included.
This model was fine-tuned by using wrime dataset.

# what is Luke?　Lukeとは？[1] 
LUKE (Language Understanding with Knowledge-based Embeddings) is a new pre-trained contextualized representation of words and entities based on transformer. LUKE treats words and entities in a given text as independent tokens, and outputs contextualized representations of them. LUKE adopts an entity-aware self-attention mechanism that is an extension of the self-attention mechanism of the transformer, and considers the types of tokens (words or entities) when computing attention scores.

LUKE achieves state-of-the-art results on five popular NLP benchmarks including SQuAD v1.1 (extractive question answering), CoNLL-2003 (named entity recognition), ReCoRD (cloze-style question answering), TACRED (relation classification), and Open Entity (entity typing).
luke-japaneseは、単語とエンティティの知識拡張型訓練済み Transformer モデルLUKEの日本語版です。LUKE は単語とエンティティを独立したトークンとして扱い、これらの文脈を考慮した表現を出力します。

# how to use 使い方
ステップ1：pythonとpytorch, sentencepieceのインストールとtransformersのアップデート（バージョンが古すぎるとLukeTokenizerが入っていないため）
update transformers and install sentencepiece, python and pytorch

ステップ2：下記のコードを実行する
Please execute this code


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
config = LukeConfig.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)    
model = AutoModelForSequenceClassification.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

text='すごく楽しかった。また行きたい。'

max_seq_length=512
token=tokenizer(text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length")
output=model(torch.tensor(token['input_ids']).unsqueeze(0), torch.tensor(token['attention_mask']).unsqueeze(0))
max_index=torch.argmax(torch.tensor(output.logits))

if max_index==0:
    print('joy、うれしい')
elif max_index==1:
    print('sadness、悲しい')
elif max_index==2:
    print('anticipation、期待')
elif max_index==3:
    print('surprise、驚き')
elif max_index==4:
    print('anger、怒り')
elif max_index==5:
    print('fear、恐れ')
elif max_index==6:
    print('disgust、嫌悪')
elif max_index==7:
    print('trust、信頼')
```

# Acknowledgments　謝辞
Lukeの開発者である山田先生とStudio ousiaさんには感謝いたします。
I would like to thank Mr.Yamada @ikuyamada and Studio ousia @StudioOusia.

# Citation
[1]@inproceedings{yamada2020luke,
  title={LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention},
  author={Ikuya Yamada and Akari Asai and Hiroyuki Shindo and Hideaki Takeda and Yuji Matsumoto},
  booktitle={EMNLP},
  year={2020}
}
