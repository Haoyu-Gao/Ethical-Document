---
language: en
license: apache-2.0
datasets:
- squad
metrics:
- squad
---

# ONNX Conversion of [distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert-base-cased-distilled-squad)

# DistilBERT base cased distilled SQuAD

This model is a fine-tune checkpoint of [DistilBERT-base-cased](https://huggingface.co/distilbert-base-cased), fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1.
This model reaches a F1 score of 87.1 on the dev set (for comparison, BERT bert-base-cased version reaches a F1 score of 88.7).
