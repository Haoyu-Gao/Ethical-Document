---
language: mn
---

# ALBERT-Mongolian
[pretraining repo link](https://github.com/bayartsogt-ya/albert-mongolian)
## Model description
Here we provide pretrained ALBERT model and trained SentencePiece model for Mongolia text. Training data is the Mongolian wikipedia corpus from Wikipedia Downloads and Mongolian News corpus.

## Evaluation Result:
```
loss = 1.7478163
masked_lm_accuracy = 0.6838185
masked_lm_loss = 1.6687671
sentence_order_accuracy = 0.998125
sentence_order_loss = 0.007942731
```

## Fine-tuning Result on Eduge Dataset:
```
                          precision    recall  f1-score   support

            байгал орчин       0.85      0.83      0.84       999
               боловсрол       0.80      0.80      0.80       873
                   спорт       0.98      0.98      0.98      2736
               технологи       0.88      0.93      0.91      1102
                 улс төр       0.92      0.85      0.89      2647
              урлаг соёл       0.93      0.94      0.94      1457
                   хууль       0.89      0.87      0.88      1651
             эдийн засаг       0.83      0.88      0.86      2509
              эрүүл мэнд       0.89      0.92      0.90      1159

                accuracy                           0.90     15133
               macro avg       0.89      0.89      0.89     15133
            weighted avg       0.90      0.90      0.90     15133
```
## Reference
1. [ALBERT - official repo](https://github.com/google-research/albert)
2. [WikiExtrator](https://github.com/attardi/wikiextractor)
3. [Mongolian BERT](https://github.com/tugstugi/mongolian-bert)
4. [ALBERT - Japanese](https://github.com/alinear-corp/albert-japanese)
5. [Mongolian Text Classification](https://github.com/sharavsambuu/mongolian-text-classification)
6. [You's paper](https://arxiv.org/abs/1904.00962)

## Citation
```
@misc{albert-mongolian,
  author = {Bayartsogt Yadamsuren},
  title = {ALBERT Pretrained Model on Mongolian Datasets},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bayartsogt-ya/albert-mongolian/}}
}
```

## For More Information
Please contact by bayartsogtyadamsuren@icloud.com
