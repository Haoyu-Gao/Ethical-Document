---
license: bsd-3-clause
---

# CodeT5+ 220M

## Model description

[CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) is a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (i.e. _encoder-only_, _decoder-only_, and _encoder-decoder_) to support a wide range of code understanding and generation tasks. 
It is introduced in the paper:

[CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf)
by [Yue Wang](https://yuewang-cuhk.github.io/)\*, [Hung Le](https://sites.google.com/view/henryle2018/home?pli=1)\*, [Akhilesh Deepak Gotmare](https://akhileshgotmare.github.io/), [Nghi D.Q. Bui](https://bdqnghi.github.io/), [Junnan Li](https://sites.google.com/site/junnanlics), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home) (* indicates equal contribution).

Compared to the original CodeT5 family (base: `220M`, large: `770M`), CodeT5+ is pretrained with a diverse set of pretraining tasks including _span denoising_, _causal language modeling_, _contrastive learning_, and _text-code matching_ to learn rich representations from both unimodal code data and bimodal code-text data. 
Additionally, it employs a simple yet effective _compute-efficient pretraining_ method to initialize the model components with frozen off-the-shelf LLMs such as [CodeGen](https://github.com/salesforce/CodeGen) to efficiently scale up the model (i.e. `2B`, `6B`, `16B`), and adopts a "shallow encoder and deep decoder" architecture. 
Furthermore, it is instruction-tuned to align with natural language instructions (see our InstructCodeT5+ 16B) following [Code Alpaca](https://github.com/sahil280114/codealpaca).  

## How to use

This model can be easily loaded using the `T5ForConditionalGeneration` functionality and employs the same tokenizer as original [CodeT5](https://github.com/salesforce/CodeT5).

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==> print "Hello World"
```

## Pretraining data

This checkpoint is trained on the stricter permissive subset of the deduplicated version of the [github-code dataset](https://huggingface.co/datasets/codeparrot/github-code).
The data is preprocessed by reserving only permissively licensed code ("mit" “apache-2”, “bsd-3-clause”, “bsd-2-clause”, “cc0-1.0”, “unlicense”, “isc”).
Supported languages (9 in total) are as follows:
`c`, `c++`, `c-sharp`,  `go`, `java`, `javascript`,  `php`, `python`, `ruby.`

## Training procedure

This checkpoint is trained on the unimodal code data at the first-stage pretraining, which includes a diverse set of pretraining tasks including _span denoising_ and two variants of _causal language modeling_.
Please refer to the paper for more details.

## Evaluation results

CodeT5+ models have been comprehensively evaluated on a wide range of code understanding and generation tasks in various settings: _zero-shot_, _finetuning_, and _instruction-tuning_.
Specifically, CodeT5+ yields substantial performance gains on many downstream tasks compared to their SoTA baselines, e.g.,
8 text-to-code retrieval tasks (+3.2 avg. MRR), 2 line-level code completion tasks (+2.1 avg. Exact Match), and 2 retrieval-augmented code generation tasks (+5.8 avg. BLEU-4). 
In 2 math programming tasks on MathQA-Python and GSM8K-Python, CodeT5+ models of below billion-parameter sizes significantly outperform many LLMs of up to 137B parameters. 
Particularly, in the zero-shot text-to-code generation task on HumanEval benchmark, InstructCodeT5+ 16B sets new SoTA results of 35.0% pass@1 and 54.5% pass@10 against other open code LLMs, even surpassing the closed-source OpenAI code-cushman-001 mode
Please refer to the [paper](https://arxiv.org/pdf/2305.07922.pdf) for more details.


## BibTeX entry and citation info

```bibtex
@article{wang2023codet5plus,
  title={CodeT5+: Open Code Large Language Models for Code Understanding and Generation},
  author={Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D.Q. and Li, Junnan and Hoi, Steven C. H.},
  journal={arXiv preprint},
  year={2023}
}
```