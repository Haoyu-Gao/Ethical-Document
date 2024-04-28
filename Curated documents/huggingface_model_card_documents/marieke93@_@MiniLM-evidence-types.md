---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: MiniLM-evidence-types
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# MiniLM-evidence-types

This model is a fine-tuned version of [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) on the evidence types dataset.
It achieved the following results on the evaluation set:
- Loss: 1.8672
- Macro f1: 0.3726
- Weighted f1: 0.7030
- Accuracy: 0.7161
- Balanced accuracy: 0.3616

## Training and evaluation data

The data set, as well as the code that was used to fine tune this model can be found in the GitHub repository [BA-Thesis-Information-Science-Persuasion-Strategies](https://github.com/mariekevdh/BA-Thesis-Information-Science-Persuasion-Strategies)

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 20
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Macro f1 | Weighted f1 | Accuracy | Balanced accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:-----------:|:--------:|:-----------------:|
| 1.4106        | 1.0   | 250  | 1.2698          | 0.1966   | 0.6084      | 0.6735   | 0.2195            |
| 1.1437        | 2.0   | 500  | 1.0985          | 0.3484   | 0.6914      | 0.7116   | 0.3536            |
| 0.9714        | 3.0   | 750  | 1.0901          | 0.2606   | 0.6413      | 0.6446   | 0.2932            |
| 0.8382        | 4.0   | 1000 | 1.0197          | 0.2764   | 0.7024      | 0.7237   | 0.2783            |
| 0.7192        | 5.0   | 1250 | 1.0895          | 0.2847   | 0.6824      | 0.6963   | 0.2915            |
| 0.6249        | 6.0   | 1500 | 1.1296          | 0.3487   | 0.6888      | 0.6948   | 0.3377            |
| 0.5336        | 7.0   | 1750 | 1.1515          | 0.3591   | 0.6982      | 0.7024   | 0.3496            |
| 0.4694        | 8.0   | 2000 | 1.1962          | 0.3626   | 0.7185      | 0.7314   | 0.3415            |
| 0.4058        | 9.0   | 2250 | 1.3313          | 0.3121   | 0.6920      | 0.7085   | 0.3033            |
| 0.3746        | 10.0  | 2500 | 1.3993          | 0.3628   | 0.6976      | 0.7047   | 0.3495            |
| 0.3267        | 11.0  | 2750 | 1.5078          | 0.3560   | 0.6958      | 0.7055   | 0.3464            |
| 0.2939        | 12.0  | 3000 | 1.5875          | 0.3685   | 0.6968      | 0.7062   | 0.3514            |
| 0.2677        | 13.0  | 3250 | 1.6470          | 0.3606   | 0.6976      | 0.7070   | 0.3490            |
| 0.2425        | 14.0  | 3500 | 1.7164          | 0.3714   | 0.7069      | 0.7207   | 0.3551            |
| 0.2301        | 15.0  | 3750 | 1.8151          | 0.3597   | 0.6975      | 0.7123   | 0.3466            |
| 0.2268        | 16.0  | 4000 | 1.7838          | 0.3940   | 0.7034      | 0.7123   | 0.3869            |
| 0.201         | 17.0  | 4250 | 1.8328          | 0.3725   | 0.6964      | 0.7062   | 0.3704            |
| 0.1923        | 18.0  | 4500 | 1.8788          | 0.3708   | 0.7019      | 0.7154   | 0.3591            |
| 0.1795        | 19.0  | 4750 | 1.8574          | 0.3752   | 0.7031      | 0.7161   | 0.3619            |
| 0.1713        | 20.0  | 5000 | 1.8672          | 0.3726   | 0.7030      | 0.7161   | 0.3616            |


### Framework versions

- Transformers 4.19.2
- Pytorch 1.11.0+cu113
- Datasets 2.2.2
- Tokenizers 0.12.1
