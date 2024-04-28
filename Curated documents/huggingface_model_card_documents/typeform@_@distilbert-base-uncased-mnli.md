---
language: en
tags:
- distilbert
datasets:
- multi_nli
metrics:
- accuracy
pipeline_tag: zero-shot-classification
---

# DistilBERT base model (uncased)


## Table of Contents
- [Model Details](#model-details)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [Environmental Impact](#environmental-impact)



## Model Details
**Model Description:**  This is the [uncased DistilBERT model](https://huggingface.co/distilbert-base-uncased) fine-tuned on [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/multi_nli) (MNLI) dataset for the zero-shot classification task. 
- **Developed by:** The [Typeform](https://www.typeform.com/) team.
- **Model Type:** Zero-Shot Classification
- **Language(s):** English
- **License:** Unknown
- **Parent Model:** See the [distilbert base uncased model](https://huggingface.co/distilbert-base-uncased) for more information about the Distilled-BERT base model.
    
   
 ## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")

model = AutoModelForSequenceClassification.from_pretrained("typeform/distilbert-base-uncased-mnli")

```

## Uses
This model can be used for text classification tasks.


## Risks, Limitations and Biases
**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).


## Training

#### Training Data


This model of DistilBERT-uncased is pretrained on the Multi-Genre Natural Language Inference [(MultiNLI)](https://huggingface.co/datasets/multi_nli) corpus. It is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information. The corpus covers a range of genres of spoken and written text, and supports a distinctive cross-genre generalization evaluation.

This model is also **not** case-sensitive, i.e., it does not make a difference between "english" and "English".


#### Training Procedure

Training is done on a [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) AWS EC2 with the following hyperparameters:

```
$ run_glue.py \
    --model_name_or_path distilbert-base-uncased \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir /tmp/distilbert-base-uncased_mnli/
```

## Evaluation


#### Evaluation Results
When fine-tuned on downstream tasks, this model achieves the following results:
- **Epoch = ** 5.0
- **Evaluation Accuracy =**  0.8206875508543532
- **Evaluation Loss =** 0.8706700205802917
- ** Evaluation Runtime = ** 17.8278
- ** Evaluation Samples per second = ** 551.498

MNLI and MNLI-mm results:

| Task | MNLI | MNLI-mm |
|:----:|:----:|:----:|
|      | 82.0 | 82.0 |



## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). We present the hardware type based on the [associated paper](https://arxiv.org/pdf/2105.09680.pdf).


**Hardware Type:** 1 NVIDIA Tesla V100 GPUs

**Hours used:**  Unknown

**Cloud Provider:** AWS  EC2 P3


**Compute Region:** Unknown



**Carbon Emitted:** (Power consumption x Time x Carbon produced based on location of power grid): Unknown

