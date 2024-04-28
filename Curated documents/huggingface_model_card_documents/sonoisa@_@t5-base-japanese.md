---
language: ja
license: cc-by-sa-4.0
tags:
- t5
- text2text-generation
- seq2seq
datasets:
- wikipedia
- oscar
- cc100
---

# 日本語T5事前学習済みモデル

This is a T5 (Text-to-Text Transfer Transformer) model pretrained on Japanese corpus.

次の日本語コーパス（約100GB）を用いて事前学習を行ったT5 (Text-to-Text Transfer Transformer) モデルです。  

* [Wikipedia](https://ja.wikipedia.org)の日本語ダンプデータ (2020年7月6日時点のもの)
* [OSCAR](https://oscar-corpus.com)の日本語コーパス
* [CC-100](http://data.statmt.org/cc-100/)の日本語コーパス

このモデルは事前学習のみを行なったものであり、特定のタスクに利用するにはファインチューニングする必要があります。  
本モデルにも、大規模コーパスを用いた言語モデルにつきまとう、学習データの内容の偏りに由来する偏った（倫理的ではなかったり、有害だったり、バイアスがあったりする）出力結果になる問題が潜在的にあります。
この問題が発生しうることを想定した上で、被害が発生しない用途にのみ利用するよう気をつけてください。

SentencePieceトークナイザーの学習には上記Wikipediaの全データを用いました。


# 転移学習のサンプルコード

https://github.com/sonoisa/t5-japanese


# ベンチマーク

## livedoorニュース分類タスク

livedoorニュースコーパスを用いたニュース記事のジャンル予測タスクの精度は次の通りです。  
Google製多言語T5モデルに比べて、モデルサイズが25%小さく、6ptほど精度が高いです。

日本語T5 ([t5-base-japanese](https://huggingface.co/sonoisa/t5-base-japanese), パラメータ数は222M, [再現用コード](https://github.com/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb))

| label       |  precision  |  recall | f1-score | support |
| ----------- | ----------- | ------- | -------- | ------- |
|           0 |      0.96   |   0.94  |    0.95  |     130 |
|           1 |      0.98   |   0.99  |    0.99  |     121 |
|           2 |      0.96   |   0.96  |    0.96  |     123 |
|           3 |      0.86   |   0.91  |    0.89  |      82 |
|           4 |      0.96   |   0.97  |    0.97  |     129 |
|           5 |      0.96   |   0.96  |    0.96  |     141 |
|           6 |      0.98   |   0.98  |    0.98  |     127 |
|           7 |      1.00   |   0.99  |    1.00  |     127 |
|           8 |      0.99   |   0.97  |    0.98  |     120 |
|   accuracy  |             |         |    0.97  |    1100 |
|  macro avg  |      0.96   |   0.96  |    0.96  |    1100 |
| weighted avg |     0.97   |   0.97  |    0.97  |    1100 |


比較対象: 多言語T5 ([google/mt5-small](https://huggingface.co/google/mt5-small), パラメータ数は300M)

| label       |  precision  |  recall | f1-score | support |
| ----------- | ----------- | ------- | -------- | ------- |
|           0 |      0.91   |   0.88  |    0.90  |     130 |
|           1 |      0.84   |   0.93  |    0.89  |     121 |
|           2 |      0.93   |   0.80  |    0.86  |     123 |
|           3 |      0.82   |   0.74  |    0.78  |      82 |
|           4 |      0.90   |   0.95  |    0.92  |     129 |
|           5 |      0.89   |   0.89  |    0.89  |     141 |
|           6 |      0.97   |   0.98  |    0.97  |     127 |
|           7 |      0.95   |   0.98  |    0.97  |     127 |
|           8 |      0.93   |   0.95  |    0.94  |     120 |
|   accuracy  |             |         |    0.91  |    1100 |
|  macro avg  |      0.91   |   0.90  |    0.90  |    1100 |
| weighted avg |     0.91   |   0.91  |    0.91  |    1100 |


## JGLUEベンチマーク

[JGLUE](https://github.com/yahoojapan/JGLUE)ベンチマークの結果は次のとおりです（順次追加）。

- MARC-ja: 準備中
- JSTS: 準備中
- JNLI: 準備中
- JSQuAD: EM=0.900, F1=0.945, [再現用コード](https://github.com/sonoisa/t5-japanese/blob/main/t5_JSQuAD.ipynb)
- JCommonsenseQA: 準備中


# 免責事項

本モデルの作者は本モデルを作成するにあたって、その内容、機能等について細心の注意を払っておりますが、モデルの出力が正確であるかどうか、安全なものであるか等について保証をするものではなく、何らの責任を負うものではありません。本モデルの利用により、万一、利用者に何らかの不都合や損害が発生したとしても、モデルやデータセットの作者や作者の所属組織は何らの責任を負うものではありません。利用者には本モデルやデータセットの作者や所属組織が責任を負わないことを明確にする義務があります。


# ライセンス

[CC-BY SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)

[Common Crawlの利用規約](http://commoncrawl.org/terms-of-use/)も守るようご注意ください。
