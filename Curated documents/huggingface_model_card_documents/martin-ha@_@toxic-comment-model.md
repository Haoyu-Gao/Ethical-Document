---
language: en
---

## Model description
This model is a fine-tuned version of the [DistilBERT model](https://huggingface.co/transformers/model_doc/distilbert.html) to classify toxic comments. 

## How to use

You can use the model with the following code.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)
print(pipeline('This is a test text.'))
```

## Limitations and Bias

This model is intended to use for classify toxic online classifications. However, one limitation of the model is that it performs poorly for some comments that mention a specific identity subgroup, like Muslim. The following table shows a evaluation score for different identity group. You can learn the specific meaning of this metrics [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation). But basically, those metrics shows how well a model performs for a specific group. The larger the number, the better.

| **subgroup**                  | **subgroup_size** | **subgroup_auc** | **bpsn_auc** | **bnsp_auc** |
| ----------------------------- | ----------------- | ---------------- | ------------ | ------------ |
| muslim                        | 108               | 0.689            | 0.811        | 0.88         |
| jewish                        | 40                | 0.749            | 0.86         | 0.825        |
| homosexual_gay_or_lesbian     | 56                | 0.795            | 0.706        | 0.972        |
| black                         | 84                | 0.866            | 0.758        | 0.975        |
| white                         | 112               | 0.876            | 0.784        | 0.97         |
| female                        | 306               | 0.898            | 0.887        | 0.948        |
| christian                     | 231               | 0.904            | 0.917        | 0.93         |
| male                          | 225               | 0.922            | 0.862        | 0.967        |
| psychiatric_or_mental_illness | 26                | 0.924            | 0.907        | 0.95         |

The table above shows that the model performs poorly for the muslim and jewish group. In fact, you pass the sentence "Muslims are people who follow or practice Islam, an Abrahamic monotheistic religion." Into the model, the model will classify it as toxic. Be mindful for this type of potential bias.

## Training data
The training data comes this [Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). We use 10% of the `train.csv` data to train the model.

## Training procedure

You can see [this documentation and codes](https://github.com/MSIA/wenyang_pan_nlp_project_2021) for how we train the model. It takes about 3 hours in a P-100 GPU.

## Evaluation results

The model achieves 94% accuracy and 0.59 f1-score in a 10000 rows held-out test set.