---
language:
- English
license: cc-by-4.0
tags:
- roberta
- roberta-base
- question-answering
- qa
datasets:
- SQuAD
---
# roberta-base + SQuAD QA

Objective:
  This is Roberta Base trained to do the SQuAD Task. This makes a QA model capable of answering questions. 
  
```
model_name = "thatdramebaazguy/roberta-base-squad"
pipeline(model=model_name, tokenizer=model_name, revision="v1.0", task="question-answering")
```

## Overview
**Language model:** roberta-base  
**Language:** English  
**Downstream-task:** QA  
**Training data:** SQuADv1  
**Eval data:** SQuAD  
**Infrastructure**: 2x Tesla v100   
**Code:**  See [example](https://github.com/adityaarunsinghal/Domain-Adaptation/blob/master/scripts/shell_scripts/train_movieR_just_squadv1.sh)    

## Hyperparameters
```
Num examples = 88567  
Num Epochs = 10
Instantaneous batch size per device = 32  
Total train batch size (w. parallel, distributed & accumulation) = 64 

``` 
## Performance

### Eval on SQuADv1
- epoch        =    10.0
- eval_samples =   10790
- exact_match  = 83.6045
- f1           = 91.1709 

### Eval on MoviesQA
- eval_samples =    5032
- exact_match = 51.6494
- f1 = 68.2615

Github Repo: 
- [Domain-Adaptation Project](https://github.com/adityaarunsinghal/Domain-Adaptation/)

---
