---
language:
- en
- ja
license: apache-2.0
library_name: transformers
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
inference: false
---
# llm-jp-1.3b-v1.0

This repository provides large language models developed by [LLM-jp](https://llm-jp.nii.ac.jp/), a collaborative project launched in Japan.

| Model Variant | 
| :--- |
|**Instruction models**|
| [llm-jp-13b-instruct-full-jaster-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-v1.0) |
| [llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0) |
| [llm-jp-13b-instruct-full-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0) |
| [llm-jp-13b-instruct-lora-jaster-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-lora-jaster-v1.0) | 
| [llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0) | 
| [llm-jp-13b-instruct-lora-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-lora-dolly-oasst-v1.0) | 


|  | 
| :--- |
|**Pre-trained models**|
| [llm-jp-13b-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-v1.0) | 
| [llm-jp-1.3b-v1.0](https://huggingface.co/llm-jp/llm-jp-1.3b-v1.0) | 
Checkpoints format: Hugging Face Transformers (Megatron-DeepSpeed format models are available [here](https://huggingface.co/llm-jp/llm-jp-13b-v1.0-mdsfmt))


## Required Libraries and Their Versions

- torch>=2.0.0
- transformers>=4.34.0
- tokenizers>=0.14.0
- accelerate==0.23.0

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-1.3b-v1.0")
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-1.3b-v1.0", device_map="auto", torch_dtype=torch.float16)
text = "自然言語処理とは何か"
tokenized_input = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        tokenized_input,
        max_new_tokens=20,
        do_sample=True,
        top_p=0.90,
        temperature=0.7,
    )[0]
print(tokenizer.decode(output))
```


## Model Details

- **Model type:** Transformer-based Language Model
- **Total seen tokens:** 300B

|Model|Params|Layers|Hidden size|Heads|Context length|
|:---:|:---:|:---:|:---:|:---:|:---:|
|13b model|13b|40|5120|40|2048|
|1.3b model|1.3b|24|2048|16|2048|


## Training

- **Pre-training:**
  - **Hardware:** 96 A100 40GB GPUs ([mdx cluster](https://mdx.jp/en/))
  - **Software:** Megatron-DeepSpeed

- **Instruction tuning:**
  - **Hardware:** 8 A100 40GB GPUs ([mdx cluster](https://mdx.jp/en/))
  - **Software:** [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), and [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## Tokenizer
The tokenizer of this model is based on [huggingface/tokenizers](https://github.com/huggingface/tokenizers) Unigram byte-fallback model.
The vocabulary entries were converted from [`llm-jp-tokenizer v2.1 (50k)`](https://github.com/llm-jp/llm-jp-tokenizer/releases/tag/v2.1).
Please refer to [README.md](https://github.com/llm-jp/llm-jp-tokenizer) of `llm-ja-tokenizer` for details on the vocabulary construction procedure.
- **Model:** Hugging Face Fast Tokenizer using Unigram byte-fallback model which requires `tokenizers>=0.14.0`
- **Training algorithm:** SentencePiece Unigram byte-fallback
- **Training data:** A subset of the datasets for model pre-training
- **Vocabulary size:** 50,570 (mixed vocabulary of Japanese, English, and source code)


## Datasets

### Pre-training

The models have been pre-trained using a blend of the following datasets.

| Language | Dataset | Tokens|
|:---:|:---:|:---:|
|Japanese|[Wikipedia](https://huggingface.co/datasets/wikipedia)|1.5B
||[mC4](https://huggingface.co/datasets/mc4)|136B
|English|[Wikipedia](https://huggingface.co/datasets/wikipedia)|5B
||[The Pile](https://huggingface.co/datasets/EleutherAI/pile)|135B
|Codes|[The Stack](https://huggingface.co/datasets/bigcode/the-stack)|10B

The pre-training was continuously conducted using a total of 10 folds of non-overlapping data, each consisting of approximately 27-28B tokens.
We finalized the pre-training with additional (potentially) high-quality 27B tokens data obtained from the identical source datasets listed above used for the 10-fold data.

### Instruction tuning

The models have been fine-tuned on the following datasets.
 
| Language | Dataset | description |
|:---|:---:|:---:|
|Japanese|[jaster](https://github.com/llm-jp/llm-jp-eval)| An automatically transformed data from the existing Japanese NLP datasets |
||[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)| A translated one by DeepL in LLM-jp |
||[OpenAssistant Conversations Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)| A translated one by DeepL in LLM-jp |


## Evaluation
You can view the evaluation results of several LLMs on this [leaderboard](http://wandb.me/llm-jp-leaderboard). We used [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval) for the evaluation.

## Risks and Limitations

The models released here are still in the early stages of our research and development and have not been tuned to ensure outputs align with human intent and safety considerations.


## Send Questions to

llm-jp(at)nii.ac.jp


## License

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)


## Model Card Authors
*The names are listed in alphabetical order.*

Hirokazu Kiyomaru, Hiroshi Matsuda, Jun Suzuki, Namgi Han, Saku Sugawara, Shota Sasaki, Shuhei Kurita, Taishi Nakamura, Takumi Okamoto.