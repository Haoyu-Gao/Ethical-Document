---
language:
- pt
license: cc-by-4.0
tags:
- named-entity-recognition
- Transformer
- pytorch
- bert
datasets:
- wiki_lingua
metrics:
- f1
- precision
- recall
widget:
- text: henrique foi no lago pescar com o pedro mais tarde foram para a casa do pedro
    fritar os peixes
- text: cinco trabalhadores da construção civil em capacetes e coletes amarelos estão
    ocupados no trabalho
- text: na quinta feira em visita a belo horizonte pedro sobrevoa a cidade atingida
    pelas chuvas
- text: coube ao representante de classe contar que na avaliação de língua portuguesa
    alguns alunos se mantiveram concentrados e outros dispersos
model-index:
- name: rpunct-ptbr
  results:
  - task:
      type: named-entity-recognition
    dataset:
      name: wiki_lingua
      type: wiki_lingua
    metrics:
    - type: f1
      value: 55.7
      name: F1 Score
    - type: precision
      value: 57.72
      name: Precision
    - type: recall
      value: 53.83
      name: Recall
---
# 🤗 bert-restore-punctuation-ptbr


* 🪄 [W&B Dashboard](https://wandb.ai/dominguesm/RestorePunctuationPTBR)
* ⛭ [GitHub](https://github.com/DominguesM/respunct)


This is a [bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) model finetuned for punctuation restoration on [WikiLingua](https://github.com/esdurmus/Wikilingua). 

This model is intended for direct use as a punctuation restoration model for the general Portuguese language. Alternatively, you can use this for further fine-tuning on domain-specific texts for punctuation restoration tasks.

Model restores the following punctuations -- **[! ? . , - : ; ' ]**

The model also restores the upper-casing of words.

-----------------------------------------------

## 🤷 Usage

🇧🇷 easy-to-use package to restore punctuation of portuguese texts.

**Below is a quick way to use the template.**

1. First, install the package.

```
pip install respunct
```

2. Sample python code.

``` python
from respunct import RestorePuncts

model = RestorePuncts()

model.restore_puncts("""
henrique foi no lago pescar com o pedro mais tarde foram para a casa do pedro fritar os peixes""")
# output:
# Henrique foi no lago pescar com o Pedro. Mais tarde, foram para a casa do Pedro fritar os peixes.

```

-----------------------------------------------
## 🎯 Accuracy

|  label                    |   precision  |  recall | f1-score  | support|
| ------------------------- | -------------|-------- | ----------|--------|
| **Upper            - OU** |      0.89    |  0.91   |   0.90    |  69376
| **None             - OO** |      0.99    |  0.98   |   0.98    | 857659
| **Full stop/period - .O** |      0.86    |  0.93   |   0.89    |  60410
| **Comma            - ,O** |      0.85    |  0.83   |   0.84    |  48608
| **Upper + Comma    - ,U** |      0.73    |  0.76   |   0.75    |   3521
| **Question         - ?O** |      0.68    |  0.78   |   0.73    |   1168
| **Upper + period   - .U** |      0.66    |  0.72   |   0.69    |   1884
| **Upper + colon    - :U** |      0.59    |  0.63   |   0.61    |    352
| **Colon            - :O** |      0.70    |  0.53   |   0.60    |   2420
| **Question Mark    - ?U** |      0.50    |  0.56   |   0.53    |     36
| **Upper + Exclam.  - !U** |      0.38    |  0.32   |   0.34    |     38
| **Exclamation Mark - !O** |      0.30    |  0.05   |   0.08    |    783
| **Semicolon        - ;O** |      0.35    |  0.04   |   0.08    |   1557
| **Apostrophe       - 'O** |      0.00    |  0.00   |   0.00    |      3
| **Hyphen           - -O** |      0.00    |  0.00   |   0.00    |      3
|                           |              |         |           |
| **accuracy**              |              |         |   0.96    | 1047818
| **macro avg**             |      0.57    |  0.54   |   0.54    | 1047818
| **weighted avg**          |      0.96    |  0.96   |   0.96    | 1047818

-----------------------------------------------


## 🤙 Contact 

[Maicon Domingues](dominguesm@outlook.com) for questions, feedback and/or requests for similar models.
