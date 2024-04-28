---
language: en
license: apache-2.0
datasets:
- conll2003
model-index:
- name: elastic/distilbert-base-cased-finetuned-conll03-english
  results:
  - task:
      type: token-classification
      name: Token Classification
    dataset:
      name: conll2003
      type: conll2003
      config: conll2003
      split: validation
    metrics:
    - type: accuracy
      value: 0.9834432212868665
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiOTZmZTJlMzUzOTAzZjg3N2UxNmMxMjQ2M2FhZTM4MDdkYzYyYTYyNjM1YjQ0M2Y4ZmIyMzkwMmY5YjZjZGVhYSIsInZlcnNpb24iOjF9.QaSLUR7AtQmE9F-h6EBueF6INQgdKwUUzS3bNvRu44rhNDY1KAJJkmDC8FeAIVMnlOSvPKvr5pOvJ59W1zckCw
    - type: precision
      value: 0.9857564461012737
      name: Precision
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMDVmNmNmNWIwNTI0Yzc0YTI1NTk2NDM4YjY4NzY0ODQ4NzQ5MDQxMzYyYWM4YzUwNmYxZWQ1NTU2YTZiM2U2MCIsInZlcnNpb24iOjF9.ui_o64VBS_oC89VeQTx_B-nUUM0ZaivFyb6wNrYZcopJXvYgzptLCkARdBKdBajFjjupdhtq1VCdGbJ3yaXgBA
    - type: recall
      value: 0.9882123948925569
      name: Recall
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODg4Mzg1NTY3NjU4ZGQxOGVhMzQxNWU0ZTYxNWM2ZTg1OGZlM2U5ZGMxYTA2NzdiZjM5YWFkZjkzOGYwYTlkMyIsInZlcnNpb24iOjF9.8jHJv_5ZQp_CX3-k8-C3c5Hs4zp7bJPRTeE5SFrNgeX-FdhPv_8bHBM_DqOD2P_nkAzQ_PtEFfEokQpouZFJCw
    - type: f1
      value: 0.9869828926905132
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYzZlOGRjMDllYWY5MjdhODk2MmNmMDk5MDQyZGYzZDYwZTE1ZDY2MDNlMzAzN2JlMmE5Y2M3ZTNkOWE2MDBjYyIsInZlcnNpb24iOjF9.VKwzPQFSbrnUZ25gkKUZvYO_xFZcaTOSkDcN-YCxksF5DRnKudKI2HmvO8l8GCsQTCoD4DiSTKzghzLMxB1jCg
    - type: loss
      value: 0.07748260349035263
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmVmOTQ2MWI2MzZhY2U2ODQ3YjA0ZWVjYzU1NGRlMTczZDI0NmM0OWI4YmIzMmEyYjlmNDIwYmRiODM4MWM0YiIsInZlcnNpb24iOjF9.0Prq087l2Xfh-ceS99zzUDcKM4Vr4CLM2rF1F1Fqd2fj9MOhVZEXF4JACVn0fWAFqfZIPS2GD8sSwfNYaXkZAA
---

[DistilBERT base cased](https://huggingface.co/distilbert-base-cased), fine-tuned for NER using the [conll03 english dataset](https://huggingface.co/datasets/conll2003). Note that this model is sensitive to capital letters — "english" is different than "English". For the case insensitive version, please use [elastic/distilbert-base-uncased-finetuned-conll03-english](https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english).

## Versions

- Transformers version: 4.3.1
- Datasets version: 1.3.0

## Training

```
$ run_ner.py \
  --model_name_or_path distilbert-base-cased \
  --label_all_tokens True \
  --return_entity_level_metrics True \
  --dataset_name conll2003 \
  --output_dir /tmp/distilbert-base-cased-finetuned-conll03-english \
  --do_train \
  --do_eval
```

After training, we update the labels to match the NER specific labels from the
dataset [conll2003](https://raw.githubusercontent.com/huggingface/datasets/1.3.0/datasets/conll2003/dataset_infos.json)
