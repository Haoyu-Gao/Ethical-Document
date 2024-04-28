---
language:
- code
license: openrail
datasets:
- bigcode/the-stack
programming_language:
- Java
- JavaScript
- Python
pipeline_tag: text-generation
inference: false
model-index:
- name: SantaCoder
  results:
  - task:
      type: text-generation
    dataset:
      name: MultiPL HumanEval (Python)
      type: nuprl/MultiPL-E
    metrics:
    - type: pass@1
      value: 0.18
      name: pass@1
      verified: false
    - type: pass@10
      value: 0.29
      name: pass@10
      verified: false
    - type: pass@100
      value: 0.49
      name: pass@100
      verified: false
    - type: pass@1
      value: 0.35
      name: pass@1
      verified: false
    - type: pass@10
      value: 0.58
      name: pass@10
      verified: false
    - type: pass@100
      value: 0.77
      name: pass@100
      verified: false
    - type: pass@1
      value: 0.16
      name: pass@1
      verified: false
    - type: pass@10
      value: 0.27
      name: pass@10
      verified: false
    - type: pass@100
      value: 0.47
      name: pass@100
      verified: false
    - type: pass@1
      value: 0.28
      name: pass@1
      verified: false
    - type: pass@10
      value: 0.51
      name: pass@10
      verified: false
    - type: pass@100
      value: 0.7
      name: pass@100
      verified: false
    - type: pass@1
      value: 0.15
      name: pass@1
      verified: false
    - type: pass@10
      value: 0.26
      name: pass@10
      verified: false
    - type: pass@100
      value: 0.41
      name: pass@100
      verified: false
    - type: pass@1
      value: 0.28
      name: pass@1
      verified: false
    - type: pass@10
      value: 0.44
      name: pass@10
      verified: false
    - type: pass@100
      value: 0.59
      name: pass@100
      verified: false
    - type: exact_match
      value: 0.62
      name: single_line
      verified: false
    - type: exact_match
      value: 0.6
      name: single_line
      verified: false
  - task:
      type: text-generation
    dataset:
      name: HumanEval FIM (Python)
      type: loubnabnl/humaneval_infilling
    metrics:
    - type: exact_match
      value: 0.44
      name: single_line
      verified: false
  - task:
      type: text-generation
    dataset:
      name: CodeXGLUE code-to-text (Python)
      type: code_x_glue_ct_code_to_text
    metrics:
    - type: bleu
      value: 18.13
      name: BLEU
      verified: false
---

# SantaCoder

![banner](https://huggingface.co/datasets/bigcode/admin/resolve/main/banner.png)

Play with the model on the [SantaCoder Space Demo](https://huggingface.co/spaces/bigcode/santacoder-demo).

#  Table of Contents

1. [Model Summary](#model-summary)
2. [Use](#use)
3. [Limitations](#limitations)
4. [Training](#training)
5. [License](#license)
6. [Citation](#citation)

# Model Summary

This is the same model as [SantaCoder](https://huggingface.co/bigcode/santacoder) but it can be loaded with transformers >=4.28.1 to use the GPTBigCode architecture.
We refer the reader to the [SantaCoder model page](https://huggingface.co/bigcode/santacoder) for full documentation about this model


- **Repository:** [bigcode/Megatron-LM](https://github.com/bigcode-project/Megatron-LM)
- **Project Website:** [bigcode-project.org](www.bigcode-project.org)
- **Paper:** [🎅SantaCoder: Don't reach for the stars!🌟](https://t.co/YV3pzUbYOr)
- **Point of Contact:** [contact@bigcode-project.org](mailto:contact@bigcode-project.org)
- **Languages:** Python, Java, and JavaScript

There are two versions (branches) of the model:
* `main`: Uses the `gpt_bigcode` model. [Requires the bigcode fork of transformers](https://github.com/bigcode-project/transformers).
* `main_custom`: Packaged with its modeling code. Requires `transformers>=4.27`.
  Alternatively, it can run on older versions by setting the configuration parameter `activation_function = "gelu_pytorch_tanh"`.

# Use

## Intended use

The model was trained on GitHub code. As such it is _not_ an instruction model and commands like "Write a function that computes the square root." do not work well.
You should phrase commands like they occur in source code such as comments (e.g. `# the following function computes the sqrt`) or write a function signature and docstring and let the model complete the function body.

### Attribution & Other Requirements

The pretraining dataset of the model was filtered for permissive licenses only. Nevertheless, the model can generate source code verbatim from the dataset. The code's license might require attribution and/or other specific requirements that must be respected. We provide a [search index](https://huggingface.co/spaces/bigcode/santacoder-search) that let's you search through the pretraining data to identify where generated code came from and apply the proper attribution to your code.

# Limitations

The model has been trained on source code in Python, Java, and JavaScript. The predominant language in source is English although other languages are also present. As such the model is capable to generate code snippets provided some context but the generated code is not guaranteed to work as intended. It can be inefficient, contain bugs or exploits.

# Training

## Model

- **Architecture:** GPT-2 model with multi-query attention and Fill-in-the-Middle objective
- **Pretraining steps:** 600K
- **Pretraining tokens:** 236 billion
- **Precision:** float16

## Hardware

- **GPUs:** 96 Tesla V100
- **Training time:** 6.2 days
- **Total FLOPS:** 2.1 x 10e21

## Software

- **Orchestration:** [Megatron-LM](https://github.com/bigcode-project/Megatron-LM)
- **Neural networks:** [PyTorch](https://github.com/pytorch/pytorch)
- **FP16 if applicable:** [apex](https://github.com/NVIDIA/apex)

# License
The model is licenses under the CodeML Open RAIL-M v0.1 license. You can find the full license [here](https://huggingface.co/spaces/bigcode/license).
