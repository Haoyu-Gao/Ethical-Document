---
language: ko
license: cc-by-nc-sa-4.0
tags:
- gpt3
---

# Ko-GPT-Trinity 1.2B (v0.5)

## Model Description

Ko-GPT-Trinity 1.2B is a transformer model designed using SK telecom's replication of the GPT-3 architecture. Ko-GPT-Trinity refers to the class of models, while 1.2B represents the number of parameters of this particular pre-trained model.

### Model date
May 2021

### Model type
Language model

### Model version
1.2 billion parameter model

## Training data

Ko-GPT-Trinity 1.2B was trained on Ko-DAT, a large scale curated dataset created by SK telecom for the purpose of training this model.

## Training procedure

This model was trained on ko-DAT for 35 billion tokens over 72,000 steps. It was trained as a masked autoregressive language model, using cross-entropy loss.

## Intended Use and Limitations

The model learns an inner representation of the Korean language that can then be used to extract features useful for downstream tasks. The model excels at generating texts from a prompt, which was the pre-training objective.

### Limitations and Biases

Ko-GPT-Trinity was trained on Ko-DAT, a dataset known to contain profanity, lewd, politically charged, and otherwise abrasive language. As such, Ko-GPT-Trinity may produce socially unacceptable text. As with all language models, it is hard to predict in advance how Ko-GPT-Trinity will respond to particular prompts and offensive content may occur without warning.

Ko-GPT-Trinity was trained as an autoregressive language model. This means that its core functionality is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, this is an active area of ongoing research. Known limitations include the following:

Predominantly Korean: Ko-GPT-Trinity was trained largely on text in the Korean language, and is best suited for classifying, searching, summarizing, or generating such text. Ko-GPT-Trinity will by default perform worse on inputs that are different from the data distribution it is trained on, including non-Korean languages as well as specific dialects of Korean that are not as well-represented in  training data. 

Interpretability & predictability: the capacity to interpret or predict how Ko-GPT-Trinity will behave is very limited, a limitation common to most deep learning systems, especially in models of this scale.

High variance on novel inputs: Ko-GPT-Trinity is not necessarily well-calibrated in its predictions on novel inputs. This can be observed in the much higher variance in its performance as compared to that of humans on standard benchmarks.

## Eval results

### Reasoning

| Model and Size          | BoolQ     | CoPA       | WiC       | 
| ----------------------- | --------- | ---------- | --------- |
| **Ko-GPT-Trinity 1.2B** | **71.77** | **68.66**  | **78.73** |
| KoElectra-base          | 65.17     | 67.56      | 77.27     |
| KoBERT-base             | 55.97     | 62.24      | 77.60     |

## Where to send questions or comments about the model
Please contact [Eric] (eric.davis@sktair.com)
