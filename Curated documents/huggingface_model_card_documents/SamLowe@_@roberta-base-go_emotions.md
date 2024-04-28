---
language: en
license: mit
tags:
- text-classification
- pytorch
- roberta
- emotions
- multi-class-classification
- multi-label-classification
datasets:
- go_emotions
widget:
- text: I am not having a great day.
---

#### Overview

Model trained from [roberta-base](https://huggingface.co/roberta-base) on the [go_emotions](https://huggingface.co/datasets/go_emotions) dataset for multi-label classification.

##### ONNX version also available

A version of this model in ONNX format (including an INT8 quantized ONNX version) is now available at [https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx). These are faster for inference, esp for smaller batch sizes, massively reduce the size of the dependencies required for inference, make inference of the model more multi-platform, and in the case of the quantized version reduce the model file/download size by 75% whilst retaining almost all the accuracy if you only need inference.

#### Dataset used for the model

[go_emotions](https://huggingface.co/datasets/go_emotions) is based on Reddit data and has 28 labels. It is a multi-label dataset where one or multiple labels may apply for any given input text, hence this model is a multi-label classification model with 28 'probability' float outputs for any given input text. Typically a threshold of 0.5 is applied to the probabilities for the prediction for each label.

#### How the model was created

The model was trained using `AutoModelForSequenceClassification.from_pretrained` with `problem_type="multi_label_classification"` for 3 epochs with a learning rate of 2e-5 and weight decay of 0.01.

#### Inference

There are multiple ways to use this model in Huggingface Transformers. Possibly the simplest is using a pipeline:

```python
from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I am not having a great day"]

model_outputs = classifier(sentences)
print(model_outputs[0])
# produces a list of dicts for each of the labels
```

#### Evaluation / metrics

Evaluation of the model is available at

- https://github.com/samlowe/go_emotions-dataset/blob/main/eval-roberta-base-go_emotions.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samlowe/go_emotions-dataset/blob/main/eval-roberta-base-go_emotions.ipynb)

##### Summary

As provided in the above notebook, evaluation of the multi-label output (of the 28 dim output via a threshold of 0.5 to binarize each) using the dataset test split gives:

- Accuracy: 0.474
- Precision: 0.575
- Recall: 0.396
- F1: 0.450

But the metrics are more meaningful when measured per label given the multi-label nature (each label is effectively an independent binary classification) and the fact that there is drastically different representations of the labels in the dataset.

With a threshold of 0.5 applied to binarize the model outputs, as per the above notebook, the metrics per label are:

|                | accuracy | precision | recall | f1    | mcc   | support | threshold |
| -------------- | -------- | --------- | ------ | ----- | ----- | ------- | --------- |
| admiration     | 0.946    | 0.725     | 0.675  | 0.699 | 0.670 | 504     | 0.5       |
| amusement      | 0.982    | 0.790     | 0.871  | 0.829 | 0.821 | 264     | 0.5       |
| anger          | 0.970    | 0.652     | 0.379  | 0.479 | 0.483 | 198     | 0.5       |
| annoyance      | 0.940    | 0.472     | 0.159  | 0.238 | 0.250 | 320     | 0.5       |
| approval       | 0.942    | 0.609     | 0.302  | 0.404 | 0.403 | 351     | 0.5       |
| caring         | 0.973    | 0.448     | 0.319  | 0.372 | 0.364 | 135     | 0.5       |
| confusion      | 0.972    | 0.500     | 0.431  | 0.463 | 0.450 | 153     | 0.5       |
| curiosity      | 0.950    | 0.537     | 0.356  | 0.428 | 0.412 | 284     | 0.5       |
| desire         | 0.987    | 0.630     | 0.410  | 0.496 | 0.502 | 83      | 0.5       |
| disappointment | 0.974    | 0.625     | 0.199  | 0.302 | 0.343 | 151     | 0.5       |
| disapproval    | 0.950    | 0.494     | 0.307  | 0.379 | 0.365 | 267     | 0.5       |
| disgust        | 0.982    | 0.707     | 0.333  | 0.453 | 0.478 | 123     | 0.5       |
| embarrassment  | 0.994    | 0.750     | 0.243  | 0.367 | 0.425 | 37      | 0.5       |
| excitement     | 0.983    | 0.603     | 0.340  | 0.435 | 0.445 | 103     | 0.5       |
| fear           | 0.992    | 0.758     | 0.603  | 0.671 | 0.672 | 78      | 0.5       |
| gratitude      | 0.990    | 0.960     | 0.881  | 0.919 | 0.914 | 352     | 0.5       |
| grief          | 0.999    | 0.000     | 0.000  | 0.000 | 0.000 | 6       | 0.5       |
| joy            | 0.978    | 0.647     | 0.559  | 0.600 | 0.590 | 161     | 0.5       |
| love           | 0.982    | 0.773     | 0.832  | 0.802 | 0.793 | 238     | 0.5       |
| nervousness    | 0.996    | 0.600     | 0.130  | 0.214 | 0.278 | 23      | 0.5       |
| optimism       | 0.972    | 0.667     | 0.376  | 0.481 | 0.488 | 186     | 0.5       |
| pride          | 0.997    | 0.000     | 0.000  | 0.000 | 0.000 | 16      | 0.5       |
| realization    | 0.974    | 0.541     | 0.138  | 0.220 | 0.264 | 145     | 0.5       |
| relief         | 0.998    | 0.000     | 0.000  | 0.000 | 0.000 | 11      | 0.5       |
| remorse        | 0.991    | 0.553     | 0.750  | 0.636 | 0.640 | 56      | 0.5       |
| sadness        | 0.977    | 0.621     | 0.494  | 0.550 | 0.542 | 156     | 0.5       |
| surprise       | 0.981    | 0.750     | 0.404  | 0.525 | 0.542 | 141     | 0.5       |
| neutral        | 0.782    | 0.694     | 0.604  | 0.646 | 0.492 | 1787    | 0.5       |

Optimizing the threshold per label for the one that gives the optimum F1 metrics gives slightly better metrics - sacrificing some precision for a greater gain in recall, hence to the benefit of F1 (how this was done is shown in the above notebook):

|                | accuracy | precision | recall | f1    | mcc   | support | threshold |
| -------------- | -------- | --------- | ------ | ----- | ----- | ------- | --------- |
| admiration     | 0.940    | 0.651     | 0.776  | 0.708 | 0.678 | 504     | 0.25      |
| amusement      | 0.982    | 0.781     | 0.890  | 0.832 | 0.825 | 264     | 0.45      |
| anger          | 0.959    | 0.454     | 0.601  | 0.517 | 0.502 | 198     | 0.15      |
| annoyance      | 0.864    | 0.243     | 0.619  | 0.349 | 0.328 | 320     | 0.10      |
| approval       | 0.926    | 0.432     | 0.442  | 0.437 | 0.397 | 351     | 0.30      |
| caring         | 0.972    | 0.426     | 0.385  | 0.405 | 0.391 | 135     | 0.40      |
| confusion      | 0.974    | 0.548     | 0.412  | 0.470 | 0.462 | 153     | 0.55      |
| curiosity      | 0.943    | 0.473     | 0.711  | 0.568 | 0.552 | 284     | 0.25      |
| desire         | 0.985    | 0.518     | 0.530  | 0.524 | 0.516 | 83      | 0.25      |
| disappointment | 0.974    | 0.562     | 0.298  | 0.390 | 0.398 | 151     | 0.40      |
| disapproval    | 0.941    | 0.414     | 0.468  | 0.439 | 0.409 | 267     | 0.30      |
| disgust        | 0.978    | 0.523     | 0.463  | 0.491 | 0.481 | 123     | 0.20      |
| embarrassment  | 0.994    | 0.567     | 0.459  | 0.507 | 0.507 | 37      | 0.10      |
| excitement     | 0.981    | 0.500     | 0.417  | 0.455 | 0.447 | 103     | 0.35      |
| fear           | 0.991    | 0.712     | 0.667  | 0.689 | 0.685 | 78      | 0.40      |
| gratitude      | 0.990    | 0.957     | 0.889  | 0.922 | 0.917 | 352     | 0.45      |
| grief          | 0.999    | 0.333     | 0.333  | 0.333 | 0.333 | 6       | 0.05      |
| joy            | 0.978    | 0.623     | 0.646  | 0.634 | 0.623 | 161     | 0.40      |
| love           | 0.982    | 0.740     | 0.899  | 0.812 | 0.807 | 238     | 0.25      |
| nervousness    | 0.996    | 0.571     | 0.348  | 0.432 | 0.444 | 23      | 0.25      |
| optimism       | 0.971    | 0.580     | 0.565  | 0.572 | 0.557 | 186     | 0.20      |
| pride          | 0.998    | 0.875     | 0.438  | 0.583 | 0.618 | 16      | 0.10      |
| realization    | 0.961    | 0.270     | 0.262  | 0.266 | 0.246 | 145     | 0.15      |
| relief         | 0.992    | 0.152     | 0.636  | 0.246 | 0.309 | 11      | 0.05      |
| remorse        | 0.991    | 0.541     | 0.946  | 0.688 | 0.712 | 56      | 0.10      |
| sadness        | 0.977    | 0.599     | 0.583  | 0.591 | 0.579 | 156     | 0.40      |
| surprise       | 0.977    | 0.543     | 0.674  | 0.601 | 0.593 | 141     | 0.15      |
| neutral        | 0.758    | 0.598     | 0.810  | 0.688 | 0.513 | 1787    | 0.25      |

This improves the overall metrics:

- Precision: 0.542
- Recall: 0.577
- F1: 0.541

Or if calculated weighted by the relative size of the support of each label:

- Precision: 0.572
- Recall: 0.677
- F1: 0.611

#### Commentary on the dataset

Some labels (E.g. gratitude) when considered independently perform very strongly with F1 exceeding 0.9, whilst others (E.g. relief) perform very poorly.

This is a challenging dataset. Labels such as relief do have much fewer examples in the training data (less than 100 out of the 40k+, and only 11 in the test split).

But there is also some ambiguity and/or labelling errors visible in the training data of go_emotions that is suspected to constrain the performance. Data cleaning on the dataset to reduce some of the mistakes, ambiguity, conflicts and duplication in the labelling would produce a higher performing model.