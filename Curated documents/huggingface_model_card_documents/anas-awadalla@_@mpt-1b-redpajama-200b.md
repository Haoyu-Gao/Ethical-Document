---
license: apache-2.0
datasets:
- togethercomputer/RedPajama-Data-1T
---

# MPT-1b-RedPajama-200b

MPT-1b-RedPajama-200b is a 1.3 billion parameter decoder-only transformer trained on the [RedPajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).
The model was trained for 200B tokens by sampling from the subsets of the RedPajama dataset in the same proportions as were used by the [Llama series of models](https://arxiv.org/abs/2302.13971).
This model was trained by [MosaicML](https://www.mosaicml.com) and follows a modified decoder-only transformer architecture.

## Model Date

April 20, 2023

## How to Use

Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. 
This is because we use a custom model architecture `MosaicGPT` that is not yet part of the `transformers` package.
`MosaicGPT` includes options for many training efficiency features such as [FlashAttention (Dao et al. 2022)](https://arxiv.org/pdf/2205.14135.pdf), [ALIBI](https://arxiv.org/abs/2108.12409), QK LayerNorm, and more.

```python
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b', trust_remote_code=True)
```

To use the optimized triton implementation of FlashAttention, you can load with `attn_impl='triton'` and move the model to `bfloat16` like so:
```python
model = transformers.AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b', trust_remote_code=True, attn_impl='triton')
model.to(device='cuda:0', dtype=torch.bfloat16)
```

## Model Description

This model uses the MosaicML LLM codebase, which can be found in the [MosaicML Examples Repository](https://github.com/mosaicml/examples/tree/v0.0.4/examples/llm).
The architecture is a modification of a standard decoder-only transformer.
The transformer has 24 layers, 16 attention heads, and width 2048.
The model has been modified from a standard transformer in the following ways:
* It uses ALiBi and does not use positional embeddings.
* It uses QK LayerNorm.
* It does not use biases.

## Training Data

The model was trained for 200B tokens (batch size 2200, sequence length 2048). It was trained on the following data mix:
* 67% RedPajama Common Crawl
* 15% [C4](https://huggingface.co/datasets/c4)
* 4.5% RedPajama GitHub
* 4.5% RedPajama Wikipedia
* 4.5% RedPajama Books
* 2.5% RedPajama Arxiv
* 2% RedPajama StackExchange

This is the same mix of data as was used in the Llama series of models](https://arxiv.org/abs/2302.13971).

Each sample was chosen from one of the datasets, with the dataset selected with the probability specified above.
The examples were shuffled within each dataset.
Each example was constructed from as many sequences from that dataset as were necessary to fill the 2048 sequence length.

The data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

## Training Configuration

This model was trained on 440 A100-40GBs for about half a day using the [MosaicML Platform](https://www.mosaicml.com/platform). The model was trained with sharded data parallelism using FSDP.

## Acknowledgements

This model builds on the work of [Together](https://www.together.xyz), which created the RedPajama dataset with the goal of mimicking the training data used to create the Llama series of models.
We gratefully acknowledge the hard work of the team that put together this dataset, and we hope this model serves as a useful companion to that work.

We also gratefully acknowledge the work of the researchers who created the Llama series of models, which was the impetus for our efforts and those who worked on the RedPajama project.
