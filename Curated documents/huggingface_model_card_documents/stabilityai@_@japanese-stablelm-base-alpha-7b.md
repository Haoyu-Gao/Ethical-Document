---
language:
- ja
license:
- apache-2.0
tags:
- japanese-stablelm
- causal-lm
datasets:
- wikipedia
- mc4
- cc100
- oscar-corpus/OSCAR-2301
- oscar-corpus/OSCAR-2201
- togethercomputer/RedPajama-Data-1T
pipeline_tag: text-generation
---

# Japanese-StableLM-Base-Alpha-7B

![japanese-stablelm-icon](./japanese-stablelm-parrot.jpg)

> "A parrot able to speak Japanese, ukiyoe, edo period" — [Stable Diffusion XL](https://clipdrop.co/stable-diffusion)

## Model Description

`japanese-stablelm-base-alpha-7b` is a 7B-parameter decoder-only language model pre-trained on a diverse collection of Japanese and English datasets which focus on maximizing Japanese language modeling performance and Japanese downstream task performance.

For an instruction-following model, check [Japanese-StableLM-Instruct-Alpha-7B](https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b) and get access by accepting the terms and conditions.

## Usage

First install additional dependencies in [requirements.txt](./requirements.txt):

```sh
pip install sentencepiece einops
```

Then start generating text with `japanese-stablelm-base-alpha-7b` by using the following code snippet:

```python
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-base-alpha-7b",
    trust_remote_code=True,
)
model.half()
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

prompt = """
AI で科学研究を加速するには、
""".strip()

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

# this is for reproducibility.
# feel free to change to get different result
seed = 23  
torch.manual_seed(seed)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=1,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)
"""
AI で科学研究を加速するには、データ駆動型文化が必要であることも明らかになってきています。研究のあらゆる側面で、データがより重要になっているのです。
20 世紀の科学は、研究者が直接研究を行うことで、研究データを活用してきました。その後、多くの科学分野ではデータは手動で分析されるようになったものの、これらの方法には多大なコストと労力がかかることが分かりました。 そこで、多くの研究者や研究者グループは、より効率的な手法を開発し、研究の規模を拡大してきました。21 世紀になると、研究者が手動で実施する必要のある研究は、その大部分を研究者が自動化できるようになりました。
"""
```

We suggest playing with different generation config (`top_p`, `repetition_penalty` etc) to find the best setup for your tasks. For example, use higher temperature for roleplay task, lower temperature for reasoning.

## Model Details

* **Model type**: `japanese-stablelm-base-alpha-7b` model is an auto-regressive language model based on the NeoX transformer architecture.
* **Language(s)**: Japanese
* **Library**: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
* **License**: This model is licensed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).


## Training

| Parameters | Hidden Size | Layers | Heads | Sequence Length |
|------------|-------------|--------|-------|-----------------|
| 7B         | 4096        | 32     | 32    | 2048            |

### Training Dataset

`japanese-stablelm-base-alpha-7b` is pre-trained on around 750B tokens from a mixture of the following corpora:

- [Japanese/English Wikipedia](https://dumps.wikimedia.org/other/cirrussearch)
- [Japanese mc4](https://huggingface.co/datasets/mc4)
- [Japanese CC-100](http://data.statmt.org/cc-100/ja.txt.xz)
- [Japanese OSCAR](https://oscar-project.github.io/documentation/)
- [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)

## Use and Limitations

### Intended Use

The model is intended to be used by all individuals as foundational models for application-specific fine-tuning without strict limitations on commercial use.

### Limitations and bias

The pre-training dataset may have contained offensive or inappropriate content even after applying data cleansing filters which can be reflected in the model generated text. We recommend users exercise reasonable caution when using these models in production systems. Do not use the model for any applications that may cause harm or distress to individuals or groups.

## Authors
- [Meng Lee](https://huggingface.co/leemeng)
- [Fujiki Nakamura](https://huggingface.co/fujiki)
- [Makoto Shing](https://huggingface.co/mkshing)
- [Paul McCann](https://huggingface.co/polm-stability)
- [Takuya Akiba](https://huggingface.co/iwiwi)
- [Naoki Orii](https://huggingface.co/mrorii)

## Acknowledgements

We are utilizing the v1 version of the [novelai-tokenizer](https://github.com/NovelAI/novelai-tokenizer), introduced by [NovelAI](https://novelai.net/), because it processes both Japanese and English text effectively and efficiently. We extend our gratitude to NovelAI for allowing us to use their remarkable work. For more details about the tokenizer, please refer to their [blog post](https://blog.novelai.net/novelais-new-llm-tokenizer-5bc140e17642).

We are grateful for the contributions of the EleutherAI Polyglot-JA team in helping us to collect a large amount of pre-training data in Japanese. Polyglot-JA members includes Hyunwoong Ko (Project Lead), Fujiki Nakamura (originally started this project when he commited to the Polyglot team), Yunho Mo, Minji Jung, KeunSeok Im, and Su-Kyeong Jang.

We are also appreciative of [AI Novelist/Sta (Bit192, Inc.)](https://ai-novel.com/index.php) and the numerous contributors from [Stable Community Japan](https://discord.gg/VPrcE475HB) for assisting us in gathering a large amount of high-quality Japanese textual data for model training.

## How to cite
```
@misc{JapaneseStableLMBaseAlpha7B, 
      url={[https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b](https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b)}, 
      title={Japanese StableLM Base Alpha 7B}, 
      author={Lee, Meng and Nakamura, Fujiki and Shing, Makoto and McCann, Paul and Akiba, Takuya and Orii, Naoki}
}
```

## Citations

```bibtext
@software{gpt-neox-library,
  title = {{GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch}},
  author = {Andonian, Alex and Anthony, Quentin and Biderman, Stella and Black, Sid and Gali, Preetham and Gao, Leo and Hallahan, Eric and Levy-Kramer, Josh and Leahy, Connor and Nestler, Lucas and Parker, Kip and Pieler, Michael and Purohit, Shivanshu and Songz, Tri and Phil, Wang and Weinbach, Samuel},
  url = {https://www.github.com/eleutherai/gpt-neox},
  doi = {10.5281/zenodo.5879544},
  month = {8},
  year = {2021},
  version = {0.0.1},
}
```