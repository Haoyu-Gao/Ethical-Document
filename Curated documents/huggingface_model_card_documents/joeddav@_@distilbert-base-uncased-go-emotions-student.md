---
language: en
license: mit
tags:
- text-classification
- pytorch
- tensorflow
datasets:
- go_emotions
widget:
- text: I feel lucky to be here.
---

# distilbert-base-uncased-go-emotions-student

## Model Description

This model is distilled from the zero-shot classification pipeline on the unlabeled GoEmotions dataset using [this
script](https://github.com/huggingface/transformers/tree/master/examples/research_projects/zero-shot-distillation).
It was trained with mixed precision for 10 epochs and otherwise used the default script arguments. 

## Intended Usage

The model can be used like any other model trained on GoEmotions, but will likely not perform as well as a model
trained with full supervision. It is primarily intended as a demo of how an expensive NLI-based zero-shot model
can be distilled to a more efficient student, allowing a classifier to be trained with only unlabeled data. Note
that although the GoEmotions dataset allow multiple labels per instance, the teacher used single-label 
classification to create psuedo-labels.
