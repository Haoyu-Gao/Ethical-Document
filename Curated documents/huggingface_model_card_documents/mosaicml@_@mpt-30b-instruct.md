---
license: cc-by-sa-3.0
tags:
- Composer
- MosaicML
- llm-foundry
datasets:
- competition_math
- conceptofmind/cot_submix_original/cot_gsm8k
- knkarthick/dialogsum
- mosaicml/dolly_hhrlhf
- duorc
- tau/scrolls/qasper
- emozilla/quality
- scrolls/summ_screen_fd
- spider
inference: false
---

# MPT-30B-Instruct

MPT-30B-Instruct is a model for short-form instruction following.
It is built by finetuning [MPT-30B](https://huggingface.co/mosaicml/mpt-30b) on [Dolly HHRLHF](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) derived from the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and the [Anthropic Helpful and Harmless (HH-RLHF)](https://huggingface.co/datasets/Anthropic/hh-rlhf) datasets. It is also trained on [Competition Math](https://huggingface.co/datasets/competition_math), [Duorc](https://huggingface.co/datasets/duorc), [CoT GSM8k](https://huggingface.co/datasets/conceptofmind/cot_submix_original), [Qasper](https://huggingface.co/datasets/allenai/qasper), [Quality](https://huggingface.co/datasets/emozilla/quality), [Summ Screen FD](https://huggingface.co/datasets/tau/scrolls) and [Spider](https://huggingface.co/datasets/spider).
  * License: _CC-By-SA-3.0_


This model was trained by [MosaicML](https://www.mosaicml.com) and follows a modified decoder-only transformer architecture.

## Model Date

June 22, 2023

## Model License

CC-By-SA-3.0

## Documentation

* [Blog post: Raising the bar for open-source foundation models](https://www.mosaicml.com/blog/mpt-30b)
* [Codebase (mosaicml/llm-foundry repo)](https://github.com/mosaicml/llm-foundry/)
* Questions: Feel free to contact us via the [MosaicML Community Slack](https://mosaicml.me/slack)!

### Example Question/Instruction

**Bespokenizer46**
> I need to convince my friend, Phyllis, that she should train a custom LLM for her Fortune 500 company using the MosaicML Platform.
> Please write an email that explains why MosaicML's emphasis on cutting edge methodology, data privacy, and efficiency are so important.
> End the email with a friendly inquiry about Phyllis's family.

**MPT-30B-Instruct**:
> Phyllis - 
> I know you're always looking for ways to improve efficiency at work and I think I've found a solution that you'll be very interested in.
> LLMs are only as good as the data they're trained on and the training process can be very time consuming and resource intensive. MosaicML uses cutting edge methodology to efficiently train high quality models without compromising data privacy.
> They also provide tools to easily connect to and use the model in your daily workflow. 
> I think you'd really enjoy speaking with their founder, we can set up a call if you're interested.
> Also, I know it's been a tough year for your family, how are things?

> Best,
> Your Friend

## How to Use

Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. This is because we use a custom model architecture that is not yet part of the `transformers` package.

It includes options for many training efficiency features such as [FlashAttention (Dao et al. 2022)](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi](https://arxiv.org/abs/2108.12409), QK LayerNorm, and more.

```python
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-30b-instruct',
  trust_remote_code=True
)
```

To use the optimized [triton implementation](https://github.com/openai/triton) of FlashAttention, you can load the model on GPU (`cuda:0`) with `attn_impl='triton'` and with `bfloat16` precision:
```python
import torch
import transformers

name = 'mosaicml/mpt-30b-instruct'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)
```

The model was trained initially on a sequence length of 2048. An additional pre-training phase was included for sequence length adaptation to 8192. However, ALiBi further enables users to increase the maximum sequence length during finetuning and/or inference. For example:

```python
import transformers

name = 'mosaicml/mpt-30b-instruct'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 16384 # (input + output) tokens can now be up to 16384

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True
)
```

This model was trained with the MPT-30B tokenizer which is based on the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer and includes additional padding and eos tokens.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-30b')
```

The model can then be used, for example, within a text-generation pipeline.  
Note: when running Torch modules in lower precision, it is best practice to use the [torch.autocast context manager](https://pytorch.org/docs/stable/amp.html).

```python
from transformers import pipeline

with torch.autocast('cuda', dtype=torch.bfloat16):
    inputs = tokenizer('Here is a recipe for vegan banana bread:\n', return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# or using the HF pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
with torch.autocast('cuda', dtype=torch.bfloat16):
    print(
        pipe('Here is a recipe for vegan banana bread:\n',
            max_new_tokens=100,
            do_sample=True,
            use_cache=True))
```

### Formatting

This model was trained on data formatted as follows:

```python
def format_prompt(instruction):
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
    return template.format(instruction=instruction)

example = "Tell me a funny joke.\nDon't make it too funny though."
fmt_ex = format_prompt(instruction=example)
```

In the above example, `fmt_ex` is ready to be tokenized and sent through the model.

## Model Description

The architecture is a modification of a standard decoder-only transformer.

The model has been modified from a standard transformer in the following ways:
* It uses [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf)
* It uses [ALiBi (Attention with Linear Biases)](https://arxiv.org/abs/2108.12409) and does not use positional embeddings
* It does not use biases


| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 29.95B |
|n_layers | 48 |
| n_heads | 64 |
| d_model | 7168 |
| vocab size | 50432 |
| sequence length | 8192 |

## Data Mix

The model was trained on the following data mix:

| Data Source | Number of Tokens in Source | Proportion |
|-------------|----------------------------|------------|
| competition_math | 1.6 M | 3.66% |
| cot_gsm8k | 3.36 M | 7.67% |
| dialogsum | 0.1 M | 0.23% |
| dolly_hhrlhf | 5.89 M | 13.43% |
| duorc | 7.8 M | 17.80% |
| qasper | 8.72 M | 19.90% |
| quality | 11.29 M | 25.78% |
| scrolls/summ_screen_fd | 4.97 M | 11.33% |
| spider | 0.089 M | 0.20% |

## PreTraining Data

For more details on the pretraining process, see [MPT-30B](https://huggingface.co/mosaicml/mpt-30b).

The data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

### Training Configuration

This model was trained on 72 A100 40GB GPUs for 8 hours using the [MosaicML Platform](https://www.mosaicml.com/platform).
The model was trained with sharded data parallelism using [FSDP](https://pytorch.org/docs/stable/fsdp.html) and used the AdamW optimizer.

## Limitations and Biases

_The following language is modified from [EleutherAI's GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)_

MPT-30B-Instruct can produce factually incorrect output, and should not be relied on to produce factually accurate information.
MPT-30B-Instruct was trained on various public datasets.
While great efforts have been taken to clean the pretraining data, it is possible that this model could generate lewd, biased or otherwise offensive outputs.


## Acknowledgements

This model was finetuned by Sam Havens, Alex Trott, and the MosaicML NLP team

## MosaicML Platform

If you're interested in [training](https://www.mosaicml.com/training) and [deploying](https://www.mosaicml.com/inference) your own MPT or LLMs on the MosaicML Platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=huggingface&utm_medium=referral&utm_campaign=mpt-30b).

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please consult an attorney before using this model for commercial purposes.

## Citation

Please cite this model using the following format:

```
@online{MosaicML2023Introducing,
    author    = {MosaicML NLP Team},
    title     = {Introducing MPT-30B: Raising the bar
for open-source foundation models},
    year      = {2023},
    url       = {www.mosaicml.com/blog/mpt-30b},
    note      = {Accessed: 2023-06-22},
    urldate   = {2023-06-22}
}
```