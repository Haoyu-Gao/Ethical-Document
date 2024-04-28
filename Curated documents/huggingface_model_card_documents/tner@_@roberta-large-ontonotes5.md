---
datasets:
- tner/ontonotes5
metrics:
- f1
- precision
- recall
pipeline_tag: token-classification
widget:
- text: Jacob Collier is a Grammy awarded artist from England.
  example_title: NER Example 1
model-index:
- name: tner/roberta-large-ontonotes5
  results:
  - task:
      type: token-classification
      name: Token Classification
    dataset:
      name: tner/ontonotes5
      type: tner/ontonotes5
      args: tner/ontonotes5
    metrics:
    - type: f1
      value: 0.908632361399938
      name: F1
    - type: precision
      value: 0.905148095909732
      name: Precision
    - type: recall
      value: 0.9121435551212579
      name: Recall
    - type: f1_macro
      value: 0.8265477704565624
      name: F1 (macro)
    - type: precision_macro
      value: 0.8170668848546687
      name: Precision (macro)
    - type: recall_macro
      value: 0.8387672780349001
      name: Recall (macro)
    - type: f1_entity_span
      value: 0.9284544931640193
      name: F1 (entity span)
    - type: precision_entity_span
      value: 0.9248942172073342
      name: Precision (entity span)
    - type: recall_entity_span
      value: 0.9320422848005685
      name: Recall (entity span)
---
# tner/roberta-large-ontonotes5

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the 
[tner/ontonotes5](https://huggingface.co/datasets/tner/ontonotes5) dataset.
Model fine-tuning is done via [T-NER](https://github.com/asahi417/tner)'s hyper-parameter search (see the repository
for more detail). It achieves the following results on the test set:
- F1 (micro): 0.908632361399938
- Precision (micro): 0.905148095909732
- Recall (micro): 0.9121435551212579
- F1 (macro): 0.8265477704565624
- Precision (macro): 0.8170668848546687
- Recall (macro): 0.8387672780349001

The per-entity breakdown of the F1 score on the test set are below:
- cardinal_number: 0.8605277329025309
- date: 0.872996300863132
- event: 0.7424242424242424
- facility: 0.7732342007434945
- geopolitical_area: 0.9687148323205043
- group: 0.9470588235294117
- language: 0.7499999999999999
- law: 0.6666666666666666
- location: 0.7593582887700535
- money: 0.901098901098901
- ordinal_number: 0.85785536159601
- organization: 0.9227360841872057
- percent: 0.9171428571428571
- person: 0.9556004036326943
- product: 0.7857142857142858
- quantity: 0.7945205479452055
- time: 0.6870588235294116
- work_of_art: 0.7151515151515151 

For F1 scores, the confidence interval is obtained by bootstrap as below:
- F1 (micro): 
    - 90%: [0.9039454247544766, 0.9128956119702822]
    - 95%: [0.9030263216115454, 0.9138350859566045] 
- F1 (macro): 
    - 90%: [0.9039454247544766, 0.9128956119702822]
    - 95%: [0.9030263216115454, 0.9138350859566045] 

Full evaluation can be found at [metric file of NER](https://huggingface.co/tner/roberta-large-ontonotes5/raw/main/eval/metric.json) 
and [metric file of entity span](https://huggingface.co/tner/roberta-large-ontonotes5/raw/main/eval/metric_span.json).

### Usage
This model can be used through the [tner library](https://github.com/asahi417/tner). Install the library via pip   
```shell
pip install tner
```
and activate model as below.
```python
from tner import TransformersNER
model = TransformersNER("tner/roberta-large-ontonotes5")
model.predict(["Jacob Collier is a Grammy awarded English artist from London"])
```
It can be used via transformers library but it is not recommended as CRF layer is not supported at the moment.

### Training hyperparameters

The following hyperparameters were used during training:
 - dataset: ['tner/ontonotes5']
 - dataset_split: train
 - dataset_name: None
 - local_dataset: None
 - model: roberta-large
 - crf: True
 - max_length: 128
 - epoch: 15
 - batch_size: 64
 - lr: 1e-05
 - random_seed: 42
 - gradient_accumulation_steps: 1
 - weight_decay: None
 - lr_warmup_step_ratio: 0.1
 - max_grad_norm: 10.0

The full configuration can be found at [fine-tuning parameter file](https://huggingface.co/tner/roberta-large-ontonotes5/raw/main/trainer_config.json).

### Reference
If you use any resource from T-NER, please consider to cite our [paper](https://aclanthology.org/2021.eacl-demos.7/).

```

@inproceedings{ushio-camacho-collados-2021-ner,
    title = "{T}-{NER}: An All-Round Python Library for Transformer-based Named Entity Recognition",
    author = "Ushio, Asahi  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-demos.7",
    doi = "10.18653/v1/2021.eacl-demos.7",
    pages = "53--62",
    abstract = "Language model (LM) pretraining has led to consistent improvements in many NLP downstream tasks, including named entity recognition (NER). In this paper, we present T-NER (Transformer-based Named Entity Recognition), a Python library for NER LM finetuning. In addition to its practical utility, T-NER facilitates the study and investigation of the cross-domain and cross-lingual generalization ability of LMs finetuned on NER. Our library also provides a web app where users can get model predictions interactively for arbitrary text, which facilitates qualitative model evaluation for non-expert programmers. We show the potential of the library by compiling nine public NER datasets into a unified format and evaluating the cross-domain and cross- lingual performance across the datasets. The results from our initial experiments show that in-domain performance is generally competitive across datasets. However, cross-domain generalization is challenging even with a large pretrained LM, which has nevertheless capacity to learn domain-specific features if fine- tuned on a combined dataset. To facilitate future research, we also release all our LM checkpoints via the Hugging Face model hub.",
}

```
