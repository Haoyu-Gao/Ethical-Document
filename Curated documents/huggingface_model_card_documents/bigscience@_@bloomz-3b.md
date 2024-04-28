---
language:
- ak
- ar
- as
- bm
- bn
- ca
- code
- en
- es
- eu
- fon
- fr
- gu
- hi
- id
- ig
- ki
- kn
- lg
- ln
- ml
- mr
- ne
- nso
- ny
- or
- pa
- pt
- rn
- rw
- sn
- st
- sw
- ta
- te
- tn
- ts
- tum
- tw
- ur
- vi
- wo
- xh
- yo
- zh
- zu
license: bigscience-bloom-rail-1.0
datasets:
- bigscience/xP3
programming_language:
- C
- C++
- C#
- Go
- Java
- JavaScript
- Lua
- PHP
- Python
- Ruby
- Rust
- Scala
- TypeScript
pipeline_tag: text-generation
widget:
- text: 一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。Would you rate the previous
    review as positive, neutral or negative?
  example_title: zh-en sentiment
- text: 一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评？
  example_title: zh-zh sentiment
- text: Suggest at least five related search terms to "Mạng neural nhân tạo".
  example_title: vi-en query
- text: Proposez au moins cinq mots clés concernant «Réseau de neurones artificiels».
  example_title: fr-fr query
- text: Explain in a sentence in Telugu what is backpropagation in neural networks.
  example_title: te-en qa
- text: Why is the sky blue?
  example_title: en-en qa
- text: 'Write a fairy tale about a troll saving a princess from a dangerous dragon.
    The fairy tale is a masterpiece that has achieved praise worldwide and its moral
    is "Heroes Come in All Shapes and Sizes". Story (in Spanish):'
  example_title: es-en fable
- text: 'Write a fable about wood elves living in a forest that is suddenly invaded
    by ogres. The fable is a masterpiece that has achieved praise worldwide and its
    moral is "Violence is the last refuge of the incompetent". Fable (in Hindi):'
  example_title: hi-en fable
model-index:
- name: bloomz-3b1
  results:
  - task:
      type: Coreference resolution
    dataset:
      name: Winogrande XL (xl)
      type: winogrande
      config: xl
      split: validation
      revision: a80f460359d1e9a67c006011c94de42a8759430c
    metrics:
    - type: Accuracy
      value: 53.67
  - task:
      type: Coreference resolution
    dataset:
      name: XWinograd (en)
      type: Muennighoff/xwinograd
      config: en
      split: test
      revision: 9dd5ea5505fad86b7bedad667955577815300cee
    metrics:
    - type: Accuracy
      value: 59.23
  - task:
      type: Coreference resolution
    dataset:
      name: XWinograd (fr)
      type: Muennighoff/xwinograd
      config: fr
      split: test
      revision: 9dd5ea5505fad86b7bedad667955577815300cee
    metrics:
    - type: Accuracy
      value: 53.01
  - task:
      type: Coreference resolution
    dataset:
      name: XWinograd (jp)
      type: Muennighoff/xwinograd
      config: jp
      split: test
      revision: 9dd5ea5505fad86b7bedad667955577815300cee
    metrics:
    - type: Accuracy
      value: 52.45
  - task:
      type: Coreference resolution
    dataset:
      name: XWinograd (pt)
      type: Muennighoff/xwinograd
      config: pt
      split: test
      revision: 9dd5ea5505fad86b7bedad667955577815300cee
    metrics:
    - type: Accuracy
      value: 53.61
  - task:
      type: Coreference resolution
    dataset:
      name: XWinograd (ru)
      type: Muennighoff/xwinograd
      config: ru
      split: test
      revision: 9dd5ea5505fad86b7bedad667955577815300cee
    metrics:
    - type: Accuracy
      value: 53.97
  - task:
      type: Coreference resolution
    dataset:
      name: XWinograd (zh)
      type: Muennighoff/xwinograd
      config: zh
      split: test
      revision: 9dd5ea5505fad86b7bedad667955577815300cee
    metrics:
    - type: Accuracy
      value: 60.91
  - task:
      type: Natural language inference
    dataset:
      name: ANLI (r1)
      type: anli
      config: r1
      split: validation
      revision: 9dbd830a06fea8b1c49d6e5ef2004a08d9f45094
    metrics:
    - type: Accuracy
      value: 40.1
  - task:
      type: Natural language inference
    dataset:
      name: ANLI (r2)
      type: anli
      config: r2
      split: validation
      revision: 9dbd830a06fea8b1c49d6e5ef2004a08d9f45094
    metrics:
    - type: Accuracy
      value: 36.8
  - task:
      type: Natural language inference
    dataset:
      name: ANLI (r3)
      type: anli
      config: r3
      split: validation
      revision: 9dbd830a06fea8b1c49d6e5ef2004a08d9f45094
    metrics:
    - type: Accuracy
      value: 40.0
  - task:
      type: Natural language inference
    dataset:
      name: SuperGLUE (cb)
      type: super_glue
      config: cb
      split: validation
      revision: 9e12063561e7e6c79099feb6d5a493142584e9e2
    metrics:
    - type: Accuracy
      value: 75.0
  - task:
      type: Natural language inference
    dataset:
      name: SuperGLUE (rte)
      type: super_glue
      config: rte
      split: validation
      revision: 9e12063561e7e6c79099feb6d5a493142584e9e2
    metrics:
    - type: Accuracy
      value: 76.17
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (ar)
      type: xnli
      config: ar
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 53.29
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (bg)
      type: xnli
      config: bg
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 43.82
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (de)
      type: xnli
      config: de
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 45.26
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (el)
      type: xnli
      config: el
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 42.61
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (en)
      type: xnli
      config: en
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 57.31
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (es)
      type: xnli
      config: es
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 56.14
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (fr)
      type: xnli
      config: fr
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 55.78
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (hi)
      type: xnli
      config: hi
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 51.49
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (ru)
      type: xnli
      config: ru
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 47.11
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (sw)
      type: xnli
      config: sw
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 47.83
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (th)
      type: xnli
      config: th
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 42.93
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (tr)
      type: xnli
      config: tr
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 37.23
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (ur)
      type: xnli
      config: ur
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 49.04
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (vi)
      type: xnli
      config: vi
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 53.98
  - task:
      type: Natural language inference
    dataset:
      name: XNLI (zh)
      type: xnli
      config: zh
      split: validation
      revision: a5a45e4ff92d5d3f34de70aaf4b72c3bdf9f7f16
    metrics:
    - type: Accuracy
      value: 54.18
  - task:
      type: Program synthesis
    dataset:
      name: HumanEval
      type: openai_humaneval
      config: None
      split: test
      revision: e8dc562f5de170c54b5481011dd9f4fa04845771
    metrics:
    - type: Pass@1
      value: 6.29
    - type: Pass@10
      value: 11.94
    - type: Pass@100
      value: 19.06
  - task:
      type: Sentence completion
    dataset:
      name: StoryCloze (2016)
      type: story_cloze
      config: '2016'
      split: validation
      revision: e724c6f8cdf7c7a2fb229d862226e15b023ee4db
    metrics:
    - type: Accuracy
      value: 87.33
  - task:
      type: Sentence completion
    dataset:
      name: SuperGLUE (copa)
      type: super_glue
      config: copa
      split: validation
      revision: 9e12063561e7e6c79099feb6d5a493142584e9e2
    metrics:
    - type: Accuracy
      value: 76.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (et)
      type: xcopa
      config: et
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 53.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (ht)
      type: xcopa
      config: ht
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 64.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (id)
      type: xcopa
      config: id
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 70.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (it)
      type: xcopa
      config: it
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 53.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (qu)
      type: xcopa
      config: qu
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 56.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (sw)
      type: xcopa
      config: sw
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 66.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (ta)
      type: xcopa
      config: ta
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 59.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (th)
      type: xcopa
      config: th
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 63.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (tr)
      type: xcopa
      config: tr
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 61.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (vi)
      type: xcopa
      config: vi
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 77.0
  - task:
      type: Sentence completion
    dataset:
      name: XCOPA (zh)
      type: xcopa
      config: zh
      split: validation
      revision: 37f73c60fb123111fa5af5f9b705d0b3747fd187
    metrics:
    - type: Accuracy
      value: 73.0
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (ar)
      type: Muennighoff/xstory_cloze
      config: ar
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 80.61
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (es)
      type: Muennighoff/xstory_cloze
      config: es
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 85.9
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (eu)
      type: Muennighoff/xstory_cloze
      config: eu
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 70.95
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (hi)
      type: Muennighoff/xstory_cloze
      config: hi
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 78.89
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (id)
      type: Muennighoff/xstory_cloze
      config: id
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 82.99
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (my)
      type: Muennighoff/xstory_cloze
      config: my
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 49.9
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (ru)
      type: Muennighoff/xstory_cloze
      config: ru
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 61.42
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (sw)
      type: Muennighoff/xstory_cloze
      config: sw
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 69.69
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (te)
      type: Muennighoff/xstory_cloze
      config: te
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 73.66
  - task:
      type: Sentence completion
    dataset:
      name: XStoryCloze (zh)
      type: Muennighoff/xstory_cloze
      config: zh
      split: validation
      revision: 8bb76e594b68147f1a430e86829d07189622b90d
    metrics:
    - type: Accuracy
      value: 84.32
---

![xmtf](https://github.com/bigscience-workshop/xmtf/blob/master/xmtf_banner.png?raw=true)

#  Table of Contents

1. [Model Summary](#model-summary)
2. [Use](#use)
3. [Limitations](#limitations)
4. [Training](#training)
5. [Evaluation](#evaluation)
7. [Citation](#citation)

# Model Summary

> We present BLOOMZ & mT0, a family of models capable of following human instructions in dozens of languages zero-shot. We finetune BLOOM & mT5 pretrained multilingual language models on our crosslingual task mixture (xP3) and find the resulting models capable of crosslingual generalization to unseen tasks & languages.

- **Repository:** [bigscience-workshop/xmtf](https://github.com/bigscience-workshop/xmtf)
- **Paper:** [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
- **Point of Contact:** [Niklas Muennighoff](mailto:niklas@hf.co)
- **Languages:** Refer to [bloom](https://huggingface.co/bigscience/bloom) for pretraining & [xP3](https://huggingface.co/datasets/bigscience/xP3) for finetuning language proportions. It understands both pretraining & finetuning languages.
- **BLOOMZ & mT0 Model Family:**

<div class="max-w-full overflow-auto">
<table>
  <tr>
<th colspan="12">Multitask finetuned on <a style="font-weight:bold" href=https://huggingface.co/datasets/bigscience/xP3>xP3</a>. Recommended for prompting in English.
</tr>
<tr>
<td>Parameters</td>
<td>300M</td>
<td>580M</td>
<td>1.2B</td>
<td>3.7B</td>
<td>13B</td>
<td>560M</td>
<td>1.1B</td>
<td>1.7B</td>
<td>3B</td>
<td>7.1B</td>
<td>176B</td>
</tr>
<tr>
<td>Finetuned Model</td>
<td><a href=https://huggingface.co/bigscience/mt0-small>mt0-small</a></td>  
<td><a href=https://huggingface.co/bigscience/mt0-base>mt0-base</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-large>mt0-large</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-xl>mt0-xl</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-xxl>mt0-xxl</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-560m>bloomz-560m</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-1b1>bloomz-1b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-1b7>bloomz-1b7</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-3b>bloomz-3b</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-7b1>bloomz-7b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz>bloomz</a></td>
</tr>
</tr>
  <tr>
<th colspan="12">Multitask finetuned on <a style="font-weight:bold" href=https://huggingface.co/datasets/bigscience/xP3mt>xP3mt</a>. Recommended for prompting in non-English.</th>
</tr>
<tr>
<td>Finetuned Model</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/mt0-xxl-mt>mt0-xxl-mt</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/bloomz-7b1-mt>bloomz-7b1-mt</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-mt>bloomz-mt</a></td>
</tr>
<th colspan="12">Multitask finetuned on <a style="font-weight:bold" href=https://huggingface.co/datasets/Muennighoff/P3>P3</a>. Released for research purposes only. Strictly inferior to above models!</th>
</tr>
<tr>
<td>Finetuned Model</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/mt0-xxl-p3>mt0-xxl-p3</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/bloomz-7b1-p3>bloomz-7b1-p3</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-p3>bloomz-p3</a></td>
</tr>
<th colspan="12">Original pretrained checkpoints. Not recommended.</th>
<tr>
<td>Pretrained Model</td>
<td><a href=https://huggingface.co/google/mt5-small>mt5-small</a></td>  
<td><a href=https://huggingface.co/google/mt5-base>mt5-base</a></td>
<td><a href=https://huggingface.co/google/mt5-large>mt5-large</a></td>
<td><a href=https://huggingface.co/google/mt5-xl>mt5-xl</a></td>
<td><a href=https://huggingface.co/google/mt5-xxl>mt5-xxl</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-560m>bloom-560m</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-1b1>bloom-1b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-1b7>bloom-1b7</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-3b>bloom-3b</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-7b1>bloom-7b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloom>bloom</a></td>
</tr>
</table>
</div>


# Use

## Intended use

We recommend using the model to perform tasks expressed in natural language. For example, given the prompt "*Translate to English: Je t’aime.*", the model will most likely answer "*I love you.*". Some prompt ideas from our paper: 
- 一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评?
- Suggest at least five related search terms to "Mạng neural nhân tạo".
- Write a fairy tale about a troll saving a princess from a dangerous dragon. The fairy tale is a masterpiece that has achieved praise worldwide and its moral is "Heroes Come in All Shapes and Sizes". Story (in Spanish):
- Explain in a sentence in Telugu what is backpropagation in neural networks.

**Feel free to share your generations in the Community tab!**

## How to use

### CPU

<details>
<summary> Click to expand </summary>

```python
# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-3b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

</details>

### GPU

<details>
<summary> Click to expand </summary>

```python
# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-3b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

</details>

### GPU in 8bit

<details>
<summary> Click to expand </summary>

```python
# pip install -q transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-3b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

</details>

<!-- Necessary for whitespace -->
###

# Limitations

**Prompt Engineering:** The performance may vary depending on the prompt. For BLOOMZ models, we recommend making it very clear when the input stops to avoid the model trying to continue it. For example, the prompt "*Translate to English: Je t'aime*" without the full stop (.) at the end, may result in the model trying to continue the French sentence. Better prompts are e.g. "*Translate to English: Je t'aime.*", "*Translate to English: Je t'aime. Translation:*" "*What is "Je t'aime." in English?*", where it is clear for the model when it should answer. Further, we recommend providing the model as much context as possible. For example, if you want it to answer in Telugu, then tell the model, e.g. "*Explain in a sentence in Telugu what is backpropagation in neural networks.*".

# Training

## Model

- **Architecture:** Same as [bloom-3b](https://huggingface.co/bigscience/bloom-3b), also refer to the `config.json` file
- **Finetuning steps:** 2000
- **Finetuning tokens:** 8.39 billion
- **Finetuning layout:** 2x pipeline parallel, 1x tensor parallel, 64x data parallel
- **Precision:** float16

## Hardware

- **CPUs:** AMD CPUs with 512GB memory per node
- **GPUs:** 128 A100 80GB GPUs with 8 GPUs per node (16 nodes) using NVLink 4 inter-gpu connects, 4 OmniPath links
- **Communication:** NCCL-communications network with a fully dedicated subnet

## Software

- **Orchestration:** [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)
- **Optimizer & parallelism:** [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- **Neural networks:** [PyTorch](https://github.com/pytorch/pytorch) (pytorch-1.11 w/ CUDA-11.5)
- **FP16 if applicable:** [apex](https://github.com/NVIDIA/apex)

# Evaluation

We refer to Table 7 from our [paper](https://arxiv.org/abs/2211.01786) & [bigscience/evaluation-results](https://huggingface.co/datasets/bigscience/evaluation-results) for zero-shot results on unseen tasks. The sidebar reports zero-shot performance of the best prompt per dataset config.

# Citation
```bibtex
@article{muennighoff2022crosslingual,
  title={Crosslingual generalization through multitask finetuning},
  author={Muennighoff, Niklas and Wang, Thomas and Sutawika, Lintang and Roberts, Adam and Biderman, Stella and Scao, Teven Le and Bari, M Saiful and Shen, Sheng and Yong, Zheng-Xin and Schoelkopf, Hailey and others},
  journal={arXiv preprint arXiv:2211.01786},
  year={2022}
}
```