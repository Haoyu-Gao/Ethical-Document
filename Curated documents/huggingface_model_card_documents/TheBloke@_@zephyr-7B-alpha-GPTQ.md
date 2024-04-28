---
language:
- en
license: mit
tags:
- generated_from_trainer
datasets:
- stingning/ultrachat
- openbmb/UltraFeedback
base_model: HuggingFaceH4/zephyr-7b-alpha
inference: false
model_creator: Hugging Face H4
model_type: mistral
prompt_template: '<|system|>

  </s>

  <|user|>

  {prompt}</s>

  <|assistant|>

  '
quantized_by: TheBloke
model-index:
- name: zephyr-7b-alpha
  results: []
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

# Zephyr 7B Alpha - GPTQ
- Model creator: [Hugging Face H4](https://huggingface.co/HuggingFaceH4)
- Original model: [Zephyr 7B Alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)

<!-- description start -->
## Description

This repo contains GPTQ model files for [Hugging Face H4's Zephyr 7B Alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/zephyr-7B-alpha-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF)
* [Hugging Face H4's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Zephyr

```
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>

```

<!-- prompt-template end -->


<!-- README_GPTQ.md-provided-files start -->
## Provided files, and GPTQ parameters

Multiple quantisation parameters are provided, to allow you to choose the best one for your hardware and requirements.

Each separate quant is in a different branch.  See below for instructions on fetching from different branches.

Most GPTQ files are made with AutoGPTQ. Mistral models are currently made with Transformers.

<details>
  <summary>Explanation of GPTQ parameters</summary>

- Bits: The bit size of the quantised model.
- GS: GPTQ group size. Higher numbers use less VRAM, but have lower quantisation accuracy. "None" is the lowest possible value.
- Act Order: True or False. Also known as `desc_act`. True results in better quantisation accuracy. Some GPTQ clients have had issues with models that use Act Order plus Group Size, but this is generally resolved now.
- Damp %: A GPTQ parameter that affects how samples are processed for quantisation. 0.01 is default, but 0.1 results in slightly better accuracy.
- GPTQ dataset: The calibration dataset used during quantisation. Using a dataset more appropriate to the model's training can improve quantisation accuracy. Note that the GPTQ calibration dataset is not the same as the dataset used to train the model - please refer to the original model repo for details of the training dataset(s).
- Sequence Length: The length of the dataset sequences used for quantisation. Ideally this is the same as the model sequence length. For some very long sequence models (16+K), a lower sequence length may have to be used. Note that a lower sequence length does not limit the sequence length of the quantised model. It only impacts the quantisation accuracy on longer inference sequences.
- ExLlama Compatibility: Whether this file can be loaded with ExLlama, which currently only supports Llama models in 4-bit.

</details>

| Branch | Bits | GS | Act Order | Damp % | GPTQ Dataset | Seq Len | Size | ExLlama | Desc |
| ------ | ---- | -- | --------- | ------ | ------------ | ------- | ---- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ/tree/main) | 4 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4095 | 4.16 GB | Yes | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. | 
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4095 | 4.57 GB | Yes | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. | 
| [gptq-8bit--1g-actorder_True](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ/tree/gptq-8bit--1g-actorder_True) | 8 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4095 | 7.52 GB | No | 8-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-8bit-128g-actorder_True](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ/tree/gptq-8bit-128g-actorder_True) | 8 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4095 | 7.68 GB | No | 8-bit, with group size 128g for higher inference quality and with Act Order for even higher accuracy. | 
| [gptq-8bit-32g-actorder_True](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ/tree/gptq-8bit-32g-actorder_True) | 8 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4095 | 8.17 GB | No | 8-bit, with group size 32g and Act Order for maximum inference quality. | 
| [gptq-4bit-64g-actorder_True](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ/tree/gptq-4bit-64g-actorder_True) | 4 | 64 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4095 | 4.29 GB | Yes | 4-bit, with Act Order and group size 64g. Uses less VRAM than 32g, but with slightly lower accuracy. |

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download, including from branches

### In text-generation-webui

To download from the `main` branch, enter `TheBloke/zephyr-7B-alpha-GPTQ` in the "Download model" box.

To download from another branch, add `:branchname` to the end of the download name, eg `TheBloke/zephyr-7B-alpha-GPTQ:gptq-4bit-32g-actorder_True`

### From the command line

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

To download the `main` branch to a folder called `zephyr-7B-alpha-GPTQ`:

```shell
mkdir zephyr-7B-alpha-GPTQ
huggingface-cli download TheBloke/zephyr-7B-alpha-GPTQ --local-dir zephyr-7B-alpha-GPTQ --local-dir-use-symlinks False
```

To download from a different branch, add the `--revision` parameter:

```shell
mkdir zephyr-7B-alpha-GPTQ
huggingface-cli download TheBloke/zephyr-7B-alpha-GPTQ --revision gptq-4bit-32g-actorder_True --local-dir zephyr-7B-alpha-GPTQ --local-dir-use-symlinks False
```

<details>
  <summary>More advanced huggingface-cli download usage</summary>

If you remove the `--local-dir-use-symlinks False` parameter, the files will instead be stored in the central Huggingface cache directory (default location on Linux is: `~/.cache/huggingface`), and symlinks will be added to the specified `--local-dir`, pointing to their real location in the cache. This allows for interrupted downloads to be resumed, and allows you to quickly clone the repo to multiple places on disk without triggering a download again. The downside, and the reason why I don't list that as the default option, is that the files are then hidden away in a cache folder and it's harder to know where your disk space is being used, and to clear it up if/when you want to remove a download model.

The cache location can be changed with the `HF_HOME` environment variable, and/or the `--cache-dir` parameter to `huggingface-cli`.

For more documentation on downloading with `huggingface-cli`, please see: [HF -> Hub Python Library -> Download files -> Download from the CLI](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli).

To accelerate downloads on fast connections (1Gbit/s or higher), install `hf_transfer`:

```shell
pip3 install hf_transfer
```

And set environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `1`:

```shell
mkdir zephyr-7B-alpha-GPTQ
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/zephyr-7B-alpha-GPTQ --local-dir zephyr-7B-alpha-GPTQ --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>

### With `git` (**not** recommended)

To clone a specific branch with `git`, use a command like this:

```shell
git clone --single-branch --branch gptq-4bit-32g-actorder_True https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ
```

Note that using Git with HF repos is strongly discouraged. It will be much slower than using `huggingface-hub`, and will use twice as much disk space as it has to store the model files twice (it stores every byte both in the intended target folder, and again in the `.git` folder as a blob.)

<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/zephyr-7B-alpha-GPTQ`.
  - To download from a specific branch, enter for example `TheBloke/zephyr-7B-alpha-GPTQ:gptq-4bit-32g-actorder_True`
  - see Provided Files above for the list of branches for each option.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `zephyr-7B-alpha-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!

<!-- README_GPTQ.md-text-generation-webui end -->

<!-- README_GPTQ.md-use-from-tgi start -->
## Serving this model from Text Generation Inference (TGI)

It's recommended to use TGI version 1.1.0 or later. The official Docker container is: `ghcr.io/huggingface/text-generation-inference:1.1.0`

Example Docker parameters:

```shell
--model-id TheBloke/zephyr-7B-alpha-GPTQ --port 3000 --quantize awq --max-input-length 3696 --max-total-tokens 4096 --max-batch-prefill-tokens 4096
```

Example Python code for interfacing with TGI (requires huggingface-hub 0.17.0 or later):

```shell
pip3 install huggingface-hub
```

```python
from huggingface_hub import InferenceClient

endpoint_url = "https://your-endpoint-url-here"

prompt = "Tell me about AI"
prompt_template=f'''<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
'''

client = InferenceClient(endpoint_url)
response = client.text_generation(prompt,
                                  max_new_tokens=128,
                                  do_sample=True,
                                  temperature=0.7,
                                  top_p=0.95,
                                  top_k=40,
                                  repetition_penalty=1.1)

print(f"Model output: {response}")
```
<!-- README_GPTQ.md-use-from-tgi end -->
<!-- README_GPTQ.md-use-from-python start -->
## How to use this GPTQ model from Python code

### Install the necessary packages

Requires: Transformers 4.33.0 or later, Optimum 1.12.0 or later, and AutoGPTQ 0.4.2 or later.

```shell
pip3 install transformers optimum
pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7
```

If you have problems installing AutoGPTQ using the pre-built wheels, install it from source instead:

```shell
pip3 uninstall -y auto-gptq
git clone https://github.com/PanQiWei/AutoGPTQ
cd AutoGPTQ
git checkout v0.4.2
pip3 install .
```

### You can then use the following code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/zephyr-7B-alpha-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

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
<!-- README_GPTQ.md-use-from-python end -->

<!-- README_GPTQ.md-compatibility start -->
## Compatibility

The files provided are tested to work with AutoGPTQ, both via Transformers and using AutoGPTQ directly. They should also work with [Occ4m's GPTQ-for-LLaMa fork](https://github.com/0cc4m/KoboldAI).

[ExLlama](https://github.com/turboderp/exllama) is compatible with Llama and Mistral models in 4-bit. Please see the Provided Files table above for per-file compatibility.

[Huggingface Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is compatible with all GPTQ models.
<!-- README_GPTQ.md-compatibility end -->

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

**Patreon special mentions**: Pierre Kircher, Stanislav Ovsiannikov, Michael Levine, Eugene Pentland, Andrey, 준교 김, Randy H, Fred von Graf, Artur Olbinski, Caitlyn Gatomon, terasurfer, Jeff Scroggin, James Bentley, Vadim, Gabriel Puliatti, Harry Royden McLaughlin, Sean Connelly, Dan Guido, Edmond Seymore, Alicia Loh, subjectnull, AzureBlack, Manuel Alberto Morcote, Thomas Belote, Lone Striker, Chris Smitley, Vitor Caleffi, Johann-Peter Hartmann, Clay Pascal, biorpg, Brandon Frisco, sidney chen, transmissions 11, Pedro Madruga, jinyuan sun, Ajan Kanaga, Emad Mostaque, Trenton Dambrowitz, Jonathan Leane, Iucharbius, usrbinkat, vamX, George Stoitzev, Luke Pendergrass, theTransient, Olakabola, Swaroop Kallakuri, Cap'n Zoog, Brandon Phillips, Michael Dempsey, Nikolai Manek, danny, Matthew Berman, Gabriel Tamborski, alfie_i, Raymond Fosdick, Tom X Nguyen, Raven Klaugh, LangChain4j, Magnesian, Illia Dulskyi, David Ziegler, Mano Prime, Luis Javier Navarrete Lozano, Erik Bjäreholt, 阿明, Nathan Dryer, Alex, Rainer Wilmers, zynix, TL, Joseph William Delisle, John Villwock, Nathan LeClaire, Willem Michiel, Joguhyik, GodLy, OG, Alps Aficionado, Jeffrey Morgan, ReadyPlayerEmma, Tiffany J. Kim, Sebastain Graf, Spencer Kim, Michael Davis, webtim, Talal Aujan, knownsqashed, John Detwiler, Imad Khwaja, Deo Leter, Jerry Meng, Elijah Stavena, Rooh Singh, Pieter, SuperWojo, Alexandros Triantafyllidis, Stephen Murray, Ai Maven, ya boyyy, Enrico Ros, Ken Nordquist, Deep Realms, Nicholas, Spiking Neurons AB, Elle, Will Dee, Jack West, RoA, Luke @flexchar, Viktor Bowallius, Derek Yates, Subspace Studios, jjj, Toran Billups, Asp the Wyvern, Fen Risland, Ilya, NimbleBox.ai, Chadd, Nitin Borwankar, Emre, Mandus, Leonard Tan, Kalila, K, Trailburnt, S_X, Cory Kujawski


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Hugging Face H4's Zephyr 7B Alpha


<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

<img src="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png" alt="Zephyr Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>


# Model Card for Zephyr 7B Alpha

Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-α is the first model in the series, and is a fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) that was trained on on a mix of publicly available, synthetic datasets using [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290). We found that removing the in-built alignment of these datasets boosted performance on [MT Bench](https://huggingface.co/spaces/lmsys/mt-bench) and made the model more helpful. However, this means that model is likely to generate problematic text when prompted to do so and should only be used for educational and research purposes.


## Model description

- **Model type:** A 7B parameter GPT-like model fine-tuned on a mix of publicly available, synthetic datasets.
- **Language(s) (NLP):** Primarily English
- **License:** MIT
- **Finetuned from model:** [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/huggingface/alignment-handbook
- **Demo:** https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat

## Intended uses & limitations

The model was initially fine-tuned on a variant of the [`UltraChat`](https://huggingface.co/datasets/stingning/ultrachat) dataset, which contains a diverse range of synthetic dialogues generated by ChatGPT. We then further aligned the model with [🤗 TRL's](https://github.com/huggingface/trl) `DPOTrainer` on the [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset, which contain 64k prompts and model completions that are ranked by GPT-4. As a result, the model can be used for chat and you can check out our [demo](https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat) to test its capabilities. 

Here's how you can run the model using the `pipeline()` function from 🤗 Transformers:

```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!
```

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Zephyr-7B-α has not been aligned to human preferences with techniques like RLHF or deployed with in-the-loop filtering of responses like ChatGPT, so the model can produce problematic outputs (especially when prompted to do so). 
It is also unknown what the size and composition of the corpus was used to train the base model (`mistralai/Mistral-7B-v0.1`), however it is likely to have included a mix of Web data and technical sources like books and code. See the [Falcon 180B model card](https://huggingface.co/tiiuae/falcon-180B#training-data) for an example of this.


## Training and evaluation data

Zephyr 7B Alpha achieves the following results on the evaluation set:

- Loss: 0.4605
- Rewards/chosen: -0.5053
- Rewards/rejected: -1.8752
- Rewards/accuracies: 0.7812
- Rewards/margins: 1.3699
- Logps/rejected: -327.4286
- Logps/chosen: -297.1040
- Logits/rejected: -2.7153
- Logits/chosen: -2.7447

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 5e-07
- train_batch_size: 2
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 16
- total_train_batch_size: 32
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rewards/chosen | Rewards/rejected | Rewards/accuracies | Rewards/margins | Logps/rejected | Logps/chosen | Logits/rejected | Logits/chosen |
|:-------------:|:-----:|:----:|:---------------:|:--------------:|:----------------:|:------------------:|:---------------:|:--------------:|:------------:|:---------------:|:-------------:|
| 0.5602        | 0.05  | 100  | 0.5589          | -0.3359        | -0.8168          | 0.7188             | 0.4809          | -306.2607      | -293.7161    | -2.6554         | -2.6797       |
| 0.4852        | 0.1   | 200  | 0.5136          | -0.5310        | -1.4994          | 0.8125             | 0.9684          | -319.9124      | -297.6181    | -2.5762         | -2.5957       |
| 0.5212        | 0.15  | 300  | 0.5168          | -0.1686        | -1.1760          | 0.7812             | 1.0074          | -313.4444      | -290.3699    | -2.6865         | -2.7125       |
| 0.5496        | 0.21  | 400  | 0.4835          | -0.1617        | -1.7170          | 0.8281             | 1.5552          | -324.2635      | -290.2326    | -2.7947         | -2.8218       |
| 0.5209        | 0.26  | 500  | 0.5054          | -0.4778        | -1.6604          | 0.7344             | 1.1826          | -323.1325      | -296.5546    | -2.8388         | -2.8667       |
| 0.4617        | 0.31  | 600  | 0.4910          | -0.3738        | -1.5180          | 0.7656             | 1.1442          | -320.2848      | -294.4741    | -2.8234         | -2.8521       |
| 0.4452        | 0.36  | 700  | 0.4838          | -0.4591        | -1.6576          | 0.7031             | 1.1986          | -323.0770      | -296.1796    | -2.7401         | -2.7653       |
| 0.4674        | 0.41  | 800  | 0.5077          | -0.5692        | -1.8659          | 0.7656             | 1.2967          | -327.2416      | -298.3818    | -2.6740         | -2.6945       |
| 0.4656        | 0.46  | 900  | 0.4927          | -0.5279        | -1.6614          | 0.7656             | 1.1335          | -323.1518      | -297.5553    | -2.7817         | -2.8015       |
| 0.4102        | 0.52  | 1000 | 0.4772          | -0.5767        | -2.0667          | 0.7656             | 1.4900          | -331.2578      | -298.5311    | -2.7160         | -2.7455       |
| 0.4663        | 0.57  | 1100 | 0.4740          | -0.8038        | -2.1018          | 0.7656             | 1.2980          | -331.9604      | -303.0741    | -2.6994         | -2.7257       |
| 0.4737        | 0.62  | 1200 | 0.4716          | -0.3783        | -1.7015          | 0.7969             | 1.3232          | -323.9545      | -294.5634    | -2.6842         | -2.7135       |
| 0.4259        | 0.67  | 1300 | 0.4866          | -0.6239        | -1.9703          | 0.7812             | 1.3464          | -329.3312      | -299.4761    | -2.7046         | -2.7356       |
| 0.4935        | 0.72  | 1400 | 0.4747          | -0.5626        | -1.7600          | 0.7812             | 1.1974          | -325.1243      | -298.2491    | -2.7153         | -2.7444       |
| 0.4211        | 0.77  | 1500 | 0.4645          | -0.6099        | -1.9993          | 0.7656             | 1.3894          | -329.9109      | -299.1959    | -2.6944         | -2.7236       |
| 0.4931        | 0.83  | 1600 | 0.4684          | -0.6798        | -2.1082          | 0.7656             | 1.4285          | -332.0890      | -300.5934    | -2.7006         | -2.7305       |
| 0.5029        | 0.88  | 1700 | 0.4595          | -0.5063        | -1.8951          | 0.7812             | 1.3889          | -327.8267      | -297.1233    | -2.7108         | -2.7403       |
| 0.4965        | 0.93  | 1800 | 0.4613          | -0.5561        | -1.9079          | 0.7812             | 1.3518          | -328.0831      | -298.1203    | -2.7226         | -2.7523       |
| 0.4337        | 0.98  | 1900 | 0.4608          | -0.5066        | -1.8718          | 0.7656             | 1.3652          | -327.3599      | -297.1296    | -2.7175         | -2.7469       |


### Framework versions

- Transformers 4.34.0
- Pytorch 2.0.1+cu118
- Datasets 2.12.0
- Tokenizers 0.14.0
