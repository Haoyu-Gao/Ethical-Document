---
language: en
license: mit
tags:
- exbert
---


# GPT-2

Test the whole generation capabilities here: https://transformer.huggingface.co/doc/gpt2-large

Pretrained model on English language using a causal language modeling (CLM) objective. It was introduced in
[this paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
and first released at [this page](https://openai.com/blog/better-language-models/).

Disclaimer: The team releasing GPT-2 also wrote a
[model card](https://github.com/openai/gpt-2/blob/master/model_card.md) for their model. Content from this model card
has been written by the Hugging Face team to complete the information they provided and give specific examples of bias.

## Model description

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This
means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots
of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely,
it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence,
shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the
predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a
prompt.

## Intended uses & limitations

You can use the raw model for text generation or fine-tune it to a downstream task. See the
[model hub](https://huggingface.co/models?filter=gpt2) to look for fine-tuned versions on a task that interests you.

### How to use

Here is how to use the ONNX models of gpt2 to get the features of a given text:

Example using transformers.pipelines:

```python
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = ORTModelForCausalLM.from_pretrained("gpt2", from_transformers=True)
onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

text = "My name is Philipp and I live in Germany."
gen = onnx_gen(text)
```

Example of text generation:

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("optimum/gpt2")
model = ORTModelForCausalLM.from_pretrained("optimum/gpt2")

inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")

gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
tokenizer.batch_decode(gen_tokens)
```
