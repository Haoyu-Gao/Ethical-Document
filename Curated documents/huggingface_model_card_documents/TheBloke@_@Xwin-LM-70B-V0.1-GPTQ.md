---
license: llama2
model_name: Xwin-LM 70B V0.1
base_model: Xwin-LM/Xwin-LM-70b-V0.1
inference: false
model_creator: Xwin-LM
model_type: llama
prompt_template: 'A chat between a curious user and an artificial intelligence assistant.
  The assistant gives helpful, detailed, and polite answers to the user''s questions.
  USER: {prompt} ASSISTANT:

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

# Xwin-LM 70B V0.1 - GPTQ
- Model creator: [Xwin-LM](https://huggingface.co/Xwin-LM)
- Original model: [Xwin-LM 70B V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-70b-V0.1)

<!-- description start -->
## Description

This repo contains GPTQ model files for [Xwin-LM's Xwin-LM 70B V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-70b-V0.1).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GGUF)
* [Xwin-LM's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/Xwin-LM/Xwin-LM-70b-V0.1)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Vicuna

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:

```

<!-- prompt-template end -->


<!-- README_GPTQ.md-provided-files start -->
## Provided files, and GPTQ parameters

Multiple quantisation parameters are provided, to allow you to choose the best one for your hardware and requirements.

Each separate quant is in a different branch.  See below for instructions on fetching from different branches.

All recent GPTQ files are made with AutoGPTQ, and all files in non-main branches are made with AutoGPTQ. Files in the `main` branch which were uploaded before August 2023 were made with GPTQ-for-LLaMa.

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
| [main](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ/tree/main) | 4 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 35.33 GB | Yes | 4-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-4bit-128g-actorder_True](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ/tree/gptq-4bit-128g-actorder_True) | 4 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 36.65 GB | Yes | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. | 
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 40.66 GB | Yes | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. | 
| [gptq-3bit--1g-actorder_True](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ/tree/gptq-3bit--1g-actorder_True) | 3 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 26.77 GB | No | 3-bit, with Act Order and no group size. Lowest possible VRAM requirements. May be lower quality than 3-bit 128g. | 
| [gptq-3bit-128g-actorder_True](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ/tree/gptq-3bit-128g-actorder_True) | 3 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 28.03 GB | No | 3-bit, with group size 128g and act-order. Higher quality than 128g-False. | 
| [gptq-3bit-32g-actorder_True](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ/tree/gptq-3bit-32g-actorder_True) | 3 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 31.84 GB | No | 3-bit, with group size 64g and act-order. Highest quality 3-bit option. |

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download, including from branches

### In text-generation-webui

To download from the `main` branch, enter `TheBloke/Xwin-LM-70B-V0.1-GPTQ` in the "Download model" box.

To download from another branch, add `:branchname` to the end of the download name, eg `TheBloke/Xwin-LM-70B-V0.1-GPTQ:gptq-4bit-128g-actorder_True`

### From the command line

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

To download the `main` branch to a folder called `Xwin-LM-70B-V0.1-GPTQ`:

```shell
mkdir Xwin-LM-70B-V0.1-GPTQ
huggingface-cli download TheBloke/Xwin-LM-70B-V0.1-GPTQ --local-dir Xwin-LM-70B-V0.1-GPTQ --local-dir-use-symlinks False
```

To download from a different branch, add the `--revision` parameter:

```shell
mkdir Xwin-LM-70B-V0.1-GPTQ
huggingface-cli download TheBloke/Xwin-LM-70B-V0.1-GPTQ --revision gptq-4bit-128g-actorder_True --local-dir Xwin-LM-70B-V0.1-GPTQ --local-dir-use-symlinks False
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
mkdir  Xwin-LM-70B-V0.1-GPTQ
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/Xwin-LM-70B-V0.1-GPTQ --local-dir Xwin-LM-70B-V0.1-GPTQ --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>

### With `git` (**not** recommended)

To clone a specific branch with `git`, use a command like this:

```shell
git clone --single-branch --branch gptq-4bit-128g-actorder_True https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ
```

Note that using Git with HF repos is strongly discouraged. It will be much slower than using `huggingface-hub`, and will use twice as much disk space as it has to store the model files twice (it stores every byte both in the intended target folder, and again in the `.git` folder as a blob.)

<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Xwin-LM-70B-V0.1-GPTQ`.
  - To download from a specific branch, enter for example `TheBloke/Xwin-LM-70B-V0.1-GPTQ:gptq-4bit-128g-actorder_True`
  - see Provided Files above for the list of branches for each option.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `Xwin-LM-70B-V0.1-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!
<!-- README_GPTQ.md-text-generation-webui end -->

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

model_name_or_path = "TheBloke/Xwin-LM-70B-V0.1-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:
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

[ExLlama](https://github.com/turboderp/exllama) is compatible with Llama models in 4-bit. Please see the Provided Files table above for per-file compatibility.

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

**Patreon special mentions**: Alicia Loh, Stephen Murray, K, Ajan Kanaga, RoA, Magnesian, Deo Leter, Olakabola, Eugene Pentland, zynix, Deep Realms, Raymond Fosdick, Elijah Stavena, Iucharbius, Erik Bjäreholt, Luis Javier Navarrete Lozano, Nicholas, theTransient, John Detwiler, alfie_i, knownsqashed, Mano Prime, Willem Michiel, Enrico Ros, LangChain4j, OG, Michael Dempsey, Pierre Kircher, Pedro Madruga, James Bentley, Thomas Belote, Luke @flexchar, Leonard Tan, Johann-Peter Hartmann, Illia Dulskyi, Fen Risland, Chadd, S_X, Jeff Scroggin, Ken Nordquist, Sean Connelly, Artur Olbinski, Swaroop Kallakuri, Jack West, Ai Maven, David Ziegler, Russ Johnson, transmissions 11, John Villwock, Alps Aficionado, Clay Pascal, Viktor Bowallius, Subspace Studios, Rainer Wilmers, Trenton Dambrowitz, vamX, Michael Levine, 준교 김, Brandon Frisco, Kalila, Trailburnt, Randy H, Talal Aujan, Nathan Dryer, Vadim, 阿明, ReadyPlayerEmma, Tiffany J. Kim, George Stoitzev, Spencer Kim, Jerry Meng, Gabriel Tamborski, Cory Kujawski, Jeffrey Morgan, Spiking Neurons AB, Edmond Seymore, Alexandros Triantafyllidis, Lone Striker, Cap'n Zoog, Nikolai Manek, danny, ya boyyy, Derek Yates, usrbinkat, Mandus, TL, Nathan LeClaire, subjectnull, Imad Khwaja, webtim, Raven Klaugh, Asp the Wyvern, Gabriel Puliatti, Caitlyn Gatomon, Joseph William Delisle, Jonathan Leane, Luke Pendergrass, SuperWojo, Sebastain Graf, Will Dee, Fred von Graf, Andrey, Dan Guido, Daniel P. Andersen, Nitin Borwankar, Elle, Vitor Caleffi, biorpg, jjj, NimbleBox.ai, Pieter, Matthew Berman, terasurfer, Michael Davis, Alex, Stanislav Ovsiannikov


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Xwin-LM's Xwin-LM 70B V0.1


<h3 align="center">
Xwin-LM: Powerful, Stable, and Reproducible LLM Alignment
</h3>

<p align="center">
  <a href="https://github.com/Xwin-LM/Xwin-LM">
    <img src="https://img.shields.io/badge/GitHub-yellow.svg?style=social&logo=github">
  </a>
  <a href="https://huggingface.co/Xwin-LM">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue">
  </a>
</p>



**Step up your LLM alignment with Xwin-LM!**

Xwin-LM aims to develop and open-source alignment technologies for large language models, including supervised fine-tuning (SFT), reward models (RM), reject sampling, reinforcement learning from human feedback (RLHF), etc. Our first release, built-upon on the Llama2 base models, ranked **TOP-1** on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). Notably, it's **the first to surpass GPT-4** on this benchmark. The project will be continuously updated.

## News

- 💥 [Sep, 2023] We released [Xwin-LM-70B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1), which has achieved a win-rate against Davinci-003 of **95.57%** on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) benchmark, ranking as **TOP-1** on AlpacaEval. **It was the FIRST model surpassing GPT-4** on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). Also note its winrate v.s. GPT-4 is **60.61**.
- 🔍 [Sep, 2023] RLHF plays crucial role in the strong performance of Xwin-LM-V0.1 release!
- 💥 [Sep, 2023] We released [Xwin-LM-13B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-13B-V0.1), which has achieved **91.76%** win-rate on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), ranking as **top-1** among all 13B models.
- 💥 [Sep, 2023] We released [Xwin-LM-7B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-7B-V0.1), which has achieved **87.82%** win-rate on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), ranking as **top-1** among all 7B models.


## Model Card
| Model        | Checkpoint | Report | License  |
|------------|------------|-------------|------------------|
|Xwin-LM-7B-V0.1| 🤗 <a href="https://huggingface.co/Xwin-LM/Xwin-LM-7B-V0.1" target="_blank">HF Link</a> | 📃**Coming soon (Stay tuned)** | <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License|
|Xwin-LM-13B-V0.1| 🤗 <a href="https://huggingface.co/Xwin-LM/Xwin-LM-13B-V0.1" target="_blank">HF Link</a> |  |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License|
|Xwin-LM-70B-V0.1| 🤗 <a href="https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1" target="_blank">HF Link</a> |  |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License|
## Benchmarks

### Xwin-LM performance on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/).

The table below displays the performance of Xwin-LM on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), where evaluates its win-rate against Text-Davinci-003 across 805 questions. To provide a comprehensive evaluation, we present, for the first time, the win-rate against ChatGPT and GPT-4 as well. Our Xwin-LM model family establish a new state-of-the-art performance across all metrics. Notably, Xwin-LM-70B-V0.1 has eclipsed GPT-4 for the first time, achieving an impressive win-rate of **95.57%** to Text-Davinci-003 and **60.61%** to GPT-4.

| **Model**                         | **AlpacaEval (winrate %)** | **AlpacaEval (winrate %)**  |**AlpacaEval (winrate %)** |
|----------------------------------|------------|----------|-------------|
|    | **v.s. Text-Davinci-003** | **v.s. ChatGPT** | **v.s. GPT4**|
| **Xwin-LM-70B-V0.1**                   | **95.57**      | **87.50**          | **60.61**            |
| GPT-4                          | 95.28  | 84.66     |  50.00      |
| WizardLM-70B-V1.0           | 92.91  | 80.19     |  46.70      |
| Llama-2-70B-Chat         | 92.66  | 81.97     |  51.19      |
| **Xwin-LM-13B-V0.1**          | **91.76**  | **81.79**  |  **55.30**    |
| ChatGPT                    | 89.37  | 50.00     |  16.60      |
| WizardLM-13B-V1.2         | 89.17  | 75.75     |  41.91      |
| **Xwin-LM-7B-V0.1**      | **87.35** | **76.40**    |  **47.57**     |
| Llama-2-13B-Chat     | 81.09  | 64.22     |  30.92      |

### Case study on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)

|   Prompts  |  GTP4 response   |  Xwin-LM response |
|----------------|------------|--------|
|I've put a plate on top of an apple, and take the plate from kitchen to dinning room. Where is the apple now|The apple is now in the dining room, as it was underneath the plate when you moved the plate from the kitchen to the dining room.|The apple is still in the kitchen, as you have only moved the plate. The apple is under the plate, so if you want to bring the apple to the dining room, you should either move the apple directly or put the apple on the plate and then move the plate to the dining room.|
| Answer the following question.<br> How do you say "good evening" in French. | In French, "good evening" is said as "bonsoir." | In French, "good evening" is translated to "bonsoir." This greeting is used in the evening and nighttime hours, typically after 6 p.m. It is a combination of "bon," which means "good," and "soir," which means "evening." To pronounce it, say "bone-swahr." |


### Xwin-LM performance on NLP foundation tasks.

The following table provides a comparison of Xwin-LMs with other LLMs on NLP foundation tasks in [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

| Model            | MMLU 5-shot | ARC 25-shot | TruthfulQA 0-shot | HellaSwag 10-shot | Average    |
|------------------|-------------|-------------|-------------------|-------------------|------------|
| Text-davinci-003 | 56.9 | **85.2**    | 59.3         | 82.2       | 70.9  |
|Vicuna-13b 1.1   | 51.3        | 53.0        | 51.8              | 80.1              | 59.1       |
|Guanaco 30B   | 57.6        | 63.7        | 50.7              | 85.1              | 64.3       |   
| WizardLM-7B 1.0      | 42.7        | 51.6        | 44.7              | 77.7              | 54.2       |
| WizardLM-13B 1.0     | 52.3        | 57.2        | 50.5              | 81.0              | 60.2       |
| WizardLM-30B 1.0    | 58.8    | 62.5 | 52.4       | 83.3          | 64.2|
| Llama-2-7B-Chat      | 48.3        | 52.9        | 45.6     | 78.6    | 56.4       |
| Llama-2-13B-Chat      | 54.6        | 59.0        | 44.1     | 81.9    | 59.9       |
| Llama-2-70B-Chat      | 63.9        | 64.6        | 52.8       | 85.9   | 66.8       |
| **Xwin-LM-7B-V0.1**      | 49.7        | 56.2        | 48.1     | 79.5    | 58.4       |
| **Xwin-LM-13B-V0.1**      | 56.6        | 62.4        | 45.5     | 83.0    | 61.9       |
| **Xwin-LM-70B-V0.1**      | **69.6**        | 70.5        | **60.1**       | **87.1**   | **71.8**       |


## Inference

### Conversation templates
To obtain desired results, please strictly follow the conversation templates when utilizing our model for inference. Our model adopts the prompt format established by [Vicuna](https://github.com/lm-sys/FastChat) and is equipped to support **multi-turn** conversations.
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi! ASSISTANT: Hello.</s>USER: Who are you? ASSISTANT: I am Xwin-LM.</s>......
```

### HuggingFace Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Xwin-LM/Xwin-LM-7B-V0.1")
tokenizer = AutoTokenizer.from_pretrained("Xwin-LM/Xwin-LM-7B-V0.1")
(
    prompt := "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: Hello, can you help me? "
            "ASSISTANT:"
)
inputs = tokenizer(prompt, return_tensors="pt")
samples = model.generate(**inputs, max_new_tokens=4096, temperature=0.7)
output = tokenizer.decode(samples[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(output) 
# Of course! I'm here to help. Please feel free to ask your question or describe the issue you're having, and I'll do my best to assist you.
```


### vllm Example
Because Xwin-LM is based on Llama2, it also offers support for rapid inference using [vllm](https://github.com/vllm-project/vllm). Please refer to [vllm](https://github.com/vllm-project/vllm) for detailed installation instructions.
```python
from vllm import LLM, SamplingParams
(
    prompt := "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: Hello, can you help me? "
            "ASSISTANT:"
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)
llm = LLM(model="Xwin-LM/Xwin-LM-7B-V0.1")
outputs = llm.generate([prompt,], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(generated_text)
```

## TODO

- [ ] Release the source code
- [ ] Release more capabilities, such as math, reasoning, and etc.

## Citation
Please consider citing our work if you use the data or code in this repo.
```
@software{xwin-lm,
  title = {Xwin-LM},
  author = {Xwin-LM Team},
  url = {https://github.com/Xwin-LM/Xwin-LM},
  version = {pre-release},
  year = {2023},
  month = {9},
}
```

## Acknowledgements

Thanks to [Llama 2](https://ai.meta.com/llama/), [FastChat](https://github.com/lm-sys/FastChat), [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm), and [vllm](https://github.com/vllm-project/vllm).
