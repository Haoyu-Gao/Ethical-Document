---
language:
- en
license: llama2
tags:
- upstage
- llama-2
- instruct
- instruction
model_name: Llama 2 70B Instruct v2
base_model: upstage/Llama-2-70b-instruct-v2
inference: false
model_creator: Upstage
model_type: llama
pipeline_tag: text-generation
prompt_template: '### System:

  {system_message}


  ### User:

  {prompt}


  ### Assistant:

  '
quantized_by: TheBloke
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# Llama 2 70B Instruct v2 - AWQ
- Model creator: [Upstage](https://huggingface.co/Upstage)
- Original model: [Llama 2 70B Instruct v2](https://huggingface.co/upstage/Llama-2-70b-instruct-v2)

<!-- description start -->
## Description

This repo contains AWQ model files for [Upstage's Llama 2 70B Instruct v2](https://huggingface.co/upstage/Llama-2-70b-instruct-v2).


### About AWQ

AWQ is an efficient, accurate and blazing-fast low-bit weight quantization method, currently supporting 4-bit quantization. Compared to GPTQ, it offers faster Transformers-based inference.

It is also now supported by continuous batching server [vLLM](https://github.com/vllm-project/vllm), allowing use of AWQ models for high-throughput concurrent inference in multi-user server scenarios. Note that, at the time of writing, overall throughput is still lower than running vLLM with unquantised models, however using AWQ enables using much smaller GPUs which can lead to easier deployment and overall cost savings. For example, a 70B model can be run on 1 x 48GB GPU instead of 2 x 80GB.
<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-GGUF)
* [Upstage's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/upstage/Llama-2-70b-instruct-v2)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Orca-Hashes

```
### System:
{system_message}

### User:
{prompt}

### Assistant:

```

<!-- prompt-template end -->


<!-- README_AWQ.md-provided-files start -->
## Provided files and AWQ parameters

For my first release of AWQ models, I am releasing 128g models only. I will consider adding 32g as well if there is interest, and once I have done perplexity and evaluation comparisons, but at this time 32g models are still not fully tested with AutoAWQ and vLLM.

Models are released as sharded safetensors files.

| Branch | Bits | GS | AWQ Dataset | Seq Len | Size |
| ------ | ---- | -- | ----------- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ/tree/main) | 4 | 128 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 36.61 GB

<!-- README_AWQ.md-provided-files end -->

<!-- README_AWQ.md-use-from-vllm start -->
## Serving this model from vLLM

Documentation on installing and using vLLM [can be found here](https://vllm.readthedocs.io/en/latest/).

- When using vLLM as a server, pass the `--quantization awq` parameter, for example:

```shell
python3 python -m vllm.entrypoints.api_server --model TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ --quantization awq
```

When using vLLM from Python code, pass the `quantization=awq` parameter, for example:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ", quantization="awq")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
<!-- README_AWQ.md-use-from-vllm start -->

<!-- README_AWQ.md-use-from-python start -->
## How to use this AWQ model from Python code

### Install the necessary packages

Requires: [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) 0.0.2 or later

```shell
pip3 install autoawq
```

If you have problems installing [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) using the pre-built wheels, install it from source instead:

```shell
pip3 uninstall -y autoawq
git clone https://github.com/casper-hansen/AutoAWQ
cd AutoAWQ
pip3 install .
```

### You can then try the following example code

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

prompt = "Tell me about AI"
prompt_template=f'''### System:
{system_message}

### User:
{prompt}

### Assistant:

'''

print("\n\n*** Generate:")

tokens = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=512
)

print("Output: ", tokenizer.decode(generation_output[0]))

# Inference can also be done using transformers' pipeline
from transformers import pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
```
<!-- README_AWQ.md-use-from-python end -->

<!-- README_AWQ.md-compatibility start -->
## Compatibility

The files provided are tested to work with [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), and [vLLM](https://github.com/vllm-project/vllm).

[Huggingface Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is not yet compatible with AWQ, but a PR is open which should bring support soon: [TGI PR #781](https://github.com/huggingface/text-generation-inference/issues/781).
<!-- README_AWQ.md-compatibility end -->

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute

Thanks to the [chirper.ai](https://chirper.ai) team!

Thanks to Clay from [gpus.llm-utils.org](llm-utils)!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Alicia Loh, Stephen Murray, K, Ajan Kanaga, RoA, Magnesian, Deo Leter, Olakabola, Eugene Pentland, zynix, Deep Realms, Raymond Fosdick, Elijah Stavena, Iucharbius, Erik Bjäreholt, Luis Javier Navarrete Lozano, Nicholas, theTransient, John Detwiler, alfie_i, knownsqashed, Mano Prime, Willem Michiel, Enrico Ros, LangChain4j, OG, Michael Dempsey, Pierre Kircher, Pedro Madruga, James Bentley, Thomas Belote, Luke @flexchar, Leonard Tan, Johann-Peter Hartmann, Illia Dulskyi, Fen Risland, Chadd, S_X, Jeff Scroggin, Ken Nordquist, Sean Connelly, Artur Olbinski, Swaroop Kallakuri, Jack West, Ai Maven, David Ziegler, Russ Johnson, transmissions 11, John Villwock, Alps Aficionado, Clay Pascal, Viktor Bowallius, Subspace Studios, Rainer Wilmers, Trenton Dambrowitz, vamX, Michael Levine, 준교 김, Brandon Frisco, Kalila, Trailburnt, Randy H, Talal Aujan, Nathan Dryer, Vadim, 阿明, ReadyPlayerEmma, Tiffany J. Kim, George Stoitzev, Spencer Kim, Jerry Meng, Gabriel Tamborski, Cory Kujawski, Jeffrey Morgan, Spiking Neurons AB, Edmond Seymore, Alexandros Triantafyllidis, Lone Striker, Cap'n Zoog, Nikolai Manek, danny, ya boyyy, Derek Yates, usrbinkat, Mandus, TL, Nathan LeClaire, subjectnull, Imad Khwaja, webtim, Raven Klaugh, Asp the Wyvern, Gabriel Puliatti, Caitlyn Gatomon, Joseph William Delisle, Jonathan Leane, Luke Pendergrass, SuperWojo, Sebastain Graf, Will Dee, Fred von Graf, Andrey, Dan Guido, Daniel P. Andersen, Nitin Borwankar, Elle, Vitor Caleffi, biorpg, jjj, NimbleBox.ai, Pieter, Matthew Berman, terasurfer, Michael Davis, Alex, Stanislav Ovsiannikov


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Upstage's Llama 2 70B Instruct v2

# Updates
Solar, a new bot created by Upstage, is now available on **Poe**. As a top-ranked model on the HuggingFace Open LLM leaderboard, and a fine tune of Llama 2, Solar is a great example of the progress enabled by open source.
Try now at https://poe.com/Solar-0-70b


# SOLAR-0-70b-16bit model card
The model name has been changed from LLaMa-2-70b-instruct-v2 to SOLAR-0-70b-16bit

## Model Details

* **Developed by**: [Upstage](https://en.upstage.ai)
* **Backbone Model**: [LLaMA-2](https://github.com/facebookresearch/llama/tree/main)
* **Language(s)**: English
* **Library**: [HuggingFace Transformers](https://github.com/huggingface/transformers)
* **License**: Fine-tuned checkpoints is licensed under the Non-Commercial Creative Commons license ([CC BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/))
* **Where to send comments**: Instructions on how to provide feedback or comments on a model can be found by opening an issue in the [Hugging Face community's model repository](https://huggingface.co/upstage/Llama-2-70b-instruct-v2/discussions)
* **Contact**: For questions and comments about the model, please email [contact@upstage.ai](mailto:contact@upstage.ai)

## Dataset Details

### Used Datasets
- Orca-style dataset
- Alpaca-style dataset
- No other dataset was used except for the dataset mentioned above
- No benchmark test set or the training set are used


### Prompt Template
```
### System:
{System}

### User:
{User}

### Assistant:
{Assistant}
```

## Usage

- The followings are tested on A100 80GB
- Our model can handle up to 10k+ input tokens, thanks to the `rope_scaling` option

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/Llama-2-70b-instruct-v2",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

prompt = "### User:\nThomas is healthy, but he has to go to the hospital. What could be the reasons?\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
del inputs["token_type_ids"]
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

## Hardware and Software

* **Hardware**: We utilized an A100x8 * 4 for training our model
* **Training Factors**: We fine-tuned this model using a combination of the [DeepSpeed library](https://github.com/microsoft/DeepSpeed) and the [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) / [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index)

## Evaluation Results

### Overview
- We conducted a performance evaluation following the tasks being evaluated on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
We evaluated our model on four benchmark datasets, which include `ARC-Challenge`, `HellaSwag`, `MMLU`, and `TruthfulQA`
We used the [lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness), specifically commit [b281b0921b636bc36ad05c0b0b0763bd6dd43463](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463).
- We used [MT-bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge), a set of challenging multi-turn open-ended questions, to evaluate the models

### Main Results
| Model | H4(Avg) | ARC | HellaSwag | MMLU | TruthfulQA | | MT_Bench |
|--------------------------------------------------------------------|----------|----------|----------|------|----------|-|-------------|
| **[Llama-2-70b-instruct-v2](https://huggingface.co/upstage/Llama-2-70b-instruct-v2)**(***Ours***, ***Open LLM Leaderboard***) | **73** | **71.1** | **87.9** | **70.6** | **62.2** | | **7.44063** |
| [Llama-2-70b-instruct](https://huggingface.co/upstage/Llama-2-70b-instruct) (Ours, Open LLM Leaderboard) | 72.3 | 70.9 | 87.5 | 69.8 | 61 | | 7.24375  |
| [llama-65b-instruct](https://huggingface.co/upstage/llama-65b-instruct) (Ours, Open LLM Leaderboard) | 69.4 | 67.6 | 86.5 | 64.9 | 58.8 | | |
| Llama-2-70b-hf | 67.3 | 67.3 | 87.3 | 69.8 | 44.9 | | |
| [llama-30b-instruct-2048](https://huggingface.co/upstage/llama-30b-instruct-2048) (Ours, Open LLM Leaderboard) | 67.0 | 64.9 | 84.9 | 61.9 | 56.3 | | |
| [llama-30b-instruct](https://huggingface.co/upstage/llama-30b-instruct) (Ours, Open LLM Leaderboard) | 65.2 | 62.5 | 86.2 | 59.4 | 52.8 | | |
| llama-65b | 64.2 | 63.5 | 86.1 | 63.9 | 43.4 | | |
| falcon-40b-instruct | 63.4 | 61.6 | 84.3 | 55.4 | 52.5 | | |

### Scripts for H4 Score Reproduction
- Prepare evaluation environments:
```
# clone the repository
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# check out the specific commit
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
# change to the repository directory
cd lm-evaluation-harness
```

## Contact Us

### About Upstage
- [Upstage](https://en.upstage.ai) is a company specialized in Large Language Models (LLMs) and AI. We will help you build private LLMs and related applications.
If you have a dataset to build domain specific LLMs or make LLM applications, please contact us at ► [click here to contact](https://www.upstage.ai/private-llm?utm_source=huggingface&utm_medium=link&utm_campaign=privatellm)
- As of August 1st, our 70B model has reached the top spot in openLLM rankings, marking itself as the current leading performer globally. 
