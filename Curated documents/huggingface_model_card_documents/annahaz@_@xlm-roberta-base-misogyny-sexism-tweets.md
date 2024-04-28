---
{}
---

This model was an experiment BUT NOT THE FINAL MODEL.

The final model was ***annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal*** (https://huggingface.co/annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal)

Please consider using/trying that model instead. 

This model was an experiment for the following paper BUT THIS MODEL IS NOT THE FINAL MODEL:
```
@InProceedings{10.1007/978-3-031-43129-6_9,
author="Chang, Rong-Ching
and May, Jonathan
and Lerman, Kristina",
editor="Thomson, Robert
and Al-khateeb, Samer
and Burger, Annetta
and Park, Patrick
and A. Pyke, Aryn",
title="Feedback Loops and Complex Dynamics of Harmful Speech in Online Discussions",
booktitle="Social, Cultural, and Behavioral Modeling",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="85--94",
abstract="Harmful and toxic speech contribute to an unwelcoming online environment that suppresses participation and conversation. Efforts have focused on detecting and mitigating harmful speech; however, the mechanisms by which toxicity degrades online discussions are not well understood. This paper makes two contributions. First, to comprehensively model harmful comments, we introduce a multilingual misogyny and sexist speech detection model (https://huggingface.co/annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal). Second, we model the complex dynamics of online discussions as feedback loops in which harmful comments lead to negative emotions which prompt even more harmful comments. To quantify the feedback loops, we use a combination of mutual Granger causality and regression to analyze discussions on two political forums on Reddit: the moderated political forum r/Politics and the moderated neutral political forum r/NeutralPolitics. Our results suggest that harmful comments and negative emotions create self-reinforcing feedback loops in forums. Contrarily, moderation with neutral discussion appears to tip interactions into self-extinguishing feedback loops that reduce harmful speech and negative emotions. Our study sheds more light on the complex dynamics of harmful speech and the role of moderation and neutral discussion in mitigating these dynamics.",
isbn="978-3-031-43129-6"
}


```


---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: xlm-roberta-base-misogyny-sexism-tweets
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# xlm-roberta-base-misogyny-sexism-tweets

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5009
- Accuracy: 0.796
- F1: 0.8132
- Precision: 0.75
- Recall: 0.888
- Mae: 0.204
- Tn: 352
- Fp: 148
- Fn: 56
- Tp: 444

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Precision | Recall | Mae   | Tn  | Fp  | Fn | Tp  |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:---------:|:------:|:-----:|:---:|:---:|:--:|:---:|
| 0.4947        | 1.0   | 1646 | 0.4683          | 0.765    | 0.7866 | 0.7205    | 0.866  | 0.235 | 332 | 168 | 67 | 433 |
| 0.4285        | 2.0   | 3292 | 0.4514          | 0.779    | 0.8004 | 0.7298    | 0.886  | 0.221 | 336 | 164 | 57 | 443 |
| 0.3721        | 3.0   | 4938 | 0.4430          | 0.781    | 0.8060 | 0.7234    | 0.91   | 0.219 | 326 | 174 | 45 | 455 |
| 0.3127        | 4.0   | 6584 | 0.5009          | 0.796    | 0.8132 | 0.75      | 0.888  | 0.204 | 352 | 148 | 56 | 444 |


### Framework versions

- Transformers 4.20.1
- Pytorch 1.12.0+cu102
- Datasets 2.3.2
- Tokenizers 0.12.1
