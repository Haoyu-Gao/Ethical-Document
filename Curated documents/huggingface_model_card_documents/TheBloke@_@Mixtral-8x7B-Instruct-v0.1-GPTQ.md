---
language:
- fr
- it
- de
- es
- en
license: apache-2.0
model_name: Mixtral 8X7B Instruct v0.1
base_model: mistralai/Mixtral-8x7B-Instruct-v0.1
inference: false
model_creator: Mistral AI_
model_type: mixtral
prompt_template: '[INST] {prompt} [/INST]

  '
quantized_by: TheBloke
widget:
- output:
    text: 'Arr, shiver me timbers! Ye have a llama on yer lawn, ye say? Well, that
      be a new one for me! Here''s what I''d suggest, arr:


      1. Firstly, ensure yer safety. Llamas may look gentle, but they can be protective
      if they feel threatened.

      2. Try to make the area less appealing to the llama. Remove any food sources
      or water that might be attracting it.

      3. Contact local animal control or a wildlife rescue organization. They be the
      experts and can provide humane ways to remove the llama from yer property.

      4. If ye have any experience with animals, you could try to gently herd the
      llama towards a nearby field or open space. But be careful, arr!


      Remember, arr, it be important to treat the llama with respect and care. It
      be a creature just trying to survive, like the rest of us.'
  text: '[INST] You are a pirate chatbot who always responds with Arr and pirate speak!

    There''s a llama on my lawn, how can I get rid of him? [/INST]'
---
<!-- markdownlint-disable MD041 -->

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

# Mixtral 8X7B Instruct v0.1 - GPTQ
- Model creator: [Mistral AI_](https://huggingface.co/mistralai)
- Original model: [Mixtral 8X7B Instruct v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

<!-- description start -->
# Description

This repo contains GPTQ model files for [Mistral AI_'s Mixtral 8X7B Instruct v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

Mixtral GPTQs currently require:
* Transformers 4.36.0 or later
* either, AutoGPTQ 0.6 compiled from source, or
* Transformers 4.37.0.dev0 compiled from Github with: `pip3 install git+https://github.com/huggingface/transformers`

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)
* [Mistral AI_'s original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Mistral

```
[INST] {prompt} [/INST]

```

<!-- prompt-template end -->



<!-- README_GPTQ.md-compatible clients start -->
## Known compatible clients / servers

GPTQ models are currently supported on Linux (NVidia/AMD) and Windows (NVidia only). macOS users: please use GGUF models.

Mixtral GPTQs currently have special requirements - see Description above.

<!-- README_GPTQ.md-compatible clients end -->

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
- ExLlama Compatibility: Whether this file can be loaded with ExLlama, which currently only supports Llama and Mistral models in 4-bit.

</details>

| Branch | Bits | GS | Act Order | Damp % | GPTQ Dataset | Seq Len | Size | ExLlama | Desc |
| ------ | ---- | -- | --------- | ------ | ------------ | ------- | ---- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/main) | 4 | None | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 23.81 GB | No | 4-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-4bit-128g-actorder_True](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-4bit-128g-actorder_True) | 4 | 128 | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 24.70 GB | No | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. | 
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 27.42 GB | No | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. | 
| [gptq-3bit--1g-actorder_True](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-3bit--1g-actorder_True) | 3 | None | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 18.01 GB | No | 3-bit, with Act Order and no group size. Lowest possible VRAM requirements. May be lower quality than 3-bit 128g. | 
| [gptq-3bit-128g-actorder_True](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-3bit-128g-actorder_True) | 3 | 128 | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 18.85 GB | No | 3-bit, with group size 128g and act-order. Higher quality than 128g-False. | 
| [gptq-8bit--1g-actorder_True](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-8bit--1g-actorder_True) | 8 | None | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 47.04 GB | No | 8-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-8bit-128g-actorder_True](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-8bit-128g-actorder_True) | 8 | 128 | Yes | 0.1 | [VMware Open Instruct](https://huggingface.co/datasets/VMware/open-instruct/viewer/) | 8192 | 48.10 GB | No | 8-bit, with group size 128g for higher inference quality and with Act Order for even higher accuracy. |

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download, including from branches

### In text-generation-webui

To download from the `main` branch, enter `TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ` in the "Download model" box.

To download from another branch, add `:branchname` to the end of the download name, eg `TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ:gptq-4bit-128g-actorder_True`

### From the command line

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

To download the `main` branch to a folder called `Mixtral-8x7B-Instruct-v0.1-GPTQ`:

```shell
mkdir Mixtral-8x7B-Instruct-v0.1-GPTQ
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ --local-dir Mixtral-8x7B-Instruct-v0.1-GPTQ --local-dir-use-symlinks False
```

To download from a different branch, add the `--revision` parameter:

```shell
mkdir Mixtral-8x7B-Instruct-v0.1-GPTQ
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ --revision gptq-4bit-128g-actorder_True --local-dir Mixtral-8x7B-Instruct-v0.1-GPTQ --local-dir-use-symlinks False
```

<details>
  <summary>More advanced huggingface-cli download usage</summary>

If you remove the `--local-dir-use-symlinks False` parameter, the files will instead be stored in the central Hugging Face cache directory (default location on Linux is: `~/.cache/huggingface`), and symlinks will be added to the specified `--local-dir`, pointing to their real location in the cache. This allows for interrupted downloads to be resumed, and allows you to quickly clone the repo to multiple places on disk without triggering a download again. The downside, and the reason why I don't list that as the default option, is that the files are then hidden away in a cache folder and it's harder to know where your disk space is being used, and to clear it up if/when you want to remove a download model.

The cache location can be changed with the `HF_HOME` environment variable, and/or the `--cache-dir` parameter to `huggingface-cli`.

For more documentation on downloading with `huggingface-cli`, please see: [HF -> Hub Python Library -> Download files -> Download from the CLI](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli).

To accelerate downloads on fast connections (1Gbit/s or higher), install `hf_transfer`:

```shell
pip3 install hf_transfer
```

And set environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `1`:

```shell
mkdir Mixtral-8x7B-Instruct-v0.1-GPTQ
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ --local-dir Mixtral-8x7B-Instruct-v0.1-GPTQ --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>

### With `git` (**not** recommended)

To clone a specific branch with `git`, use a command like this:

```shell
git clone --single-branch --branch gptq-4bit-128g-actorder_True https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
```

Note that using Git with HF repos is strongly discouraged. It will be much slower than using `huggingface-hub`, and will use twice as much disk space as it has to store the model files twice (it stores every byte both in the intended target folder, and again in the `.git` folder as a blob.)

<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

**NOTE**: Requires:

* Transformers 4.36.0, or Transformers 4.37.0.dev0 from Github
* Either AutoGPTQ 0.6 compiled from source and `Loader: AutoGPTQ`,
* or, `Loader: Transformers`, if you installed Transformers from Github: `pip3 install git+https://github.com/huggingface/transformers`

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ`.

    - To download from a specific branch, enter for example `TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ:gptq-4bit-128g-actorder_True`
    - see Provided Files above for the list of branches for each option.

3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `Mixtral-8x7B-Instruct-v0.1-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.

    - Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.

9. Once you're ready, click the **Text Generation** tab and enter a prompt to get started!

<!-- README_GPTQ.md-text-generation-webui end -->

<!-- README_GPTQ.md-use-from-tgi start -->
## Serving this model from Text Generation Inference (TGI)

Not currently supported for Mixtral models.

<!-- README_GPTQ.md-use-from-tgi end -->
<!-- README_GPTQ.md-use-from-python start -->
## Python code example: inference from this GPTQ model

### Install the necessary packages

Requires: Transformers 4.37.0.dev0 from Github, Optimum 1.16.0 or later, and AutoGPTQ 0.5.1 or later.

```shell
pip3 install --upgrade "git+https://github.com/huggingface/transformers" optimum
# If using PyTorch 2.1 + CUDA 12.x:
pip3 install --upgrade auto-gptq
# or, if using PyTorch 2.1 + CUDA 11.x:
pip3 install --upgrade auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

If you are using PyTorch 2.0, you will need to install AutoGPTQ from source. Likewise if you have problems with the pre-built wheels, you should try building from source:

```shell
pip3 uninstall -y auto-gptq
git clone https://github.com/PanQiWei/AutoGPTQ
cd AutoGPTQ
DISABLE_QIGEN=1 pip3 install .
```

### Example Python code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Write a story about llamas"
system_message = "You are a story writing assistant"
prompt_template=f'''[INST] {prompt} [/INST]
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

The files provided are tested to work with AutoGPTQ 0.6 (compiled from source) and Transformers 4.37.0 (installed from Github).

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

**Patreon special mentions**: Michael Levine, 阿明, Trailburnt, Nikolai Manek, John Detwiler, Randy H, Will Dee, Sebastain Graf, NimbleBox.ai, Eugene Pentland, Emad Mostaque, Ai Maven, Jim Angel, Jeff Scroggin, Michael Davis, Manuel Alberto Morcote, Stephen Murray, Robert, Justin Joy, Luke @flexchar, Brandon Frisco, Elijah Stavena, S_X, Dan Guido, Undi ., Komninos Chatzipapas, Shadi, theTransient, Lone Striker, Raven Klaugh, jjj, Cap'n Zoog, Michel-Marie MAUDET (LINAGORA), Matthew Berman, David, Fen Risland, Omer Bin Jawed, Luke Pendergrass, Kalila, OG, Erik Bjäreholt, Rooh Singh, Joseph William Delisle, Dan Lewis, TL, John Villwock, AzureBlack, Brad, Pedro Madruga, Caitlyn Gatomon, K, jinyuan sun, Mano Prime, Alex, Jeffrey Morgan, Alicia Loh, Illia Dulskyi, Chadd, transmissions 11, fincy, Rainer Wilmers, ReadyPlayerEmma, knownsqashed, Mandus, biorpg, Deo Leter, Brandon Phillips, SuperWojo, Sean Connelly, Iucharbius, Jack West, Harry Royden McLaughlin, Nicholas, terasurfer, Vitor Caleffi, Duane Dunston, Johann-Peter Hartmann, David Ziegler, Olakabola, Ken Nordquist, Trenton Dambrowitz, Tom X Nguyen, Vadim, Ajan Kanaga, Leonard Tan, Clay Pascal, Alexandros Triantafyllidis, JM33133, Xule, vamX, ya boyyy, subjectnull, Talal Aujan, Alps Aficionado, wassieverse, Ari Malik, James Bentley, Woland, Spencer Kim, Michael Dempsey, Fred von Graf, Elle, zynix, William Richards, Stanislav Ovsiannikov, Edmond Seymore, Jonathan Leane, Martin Kemka, usrbinkat, Enrico Ros


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Mistral AI_'s Mixtral 8X7B Instruct v0.1

# Model Card for Mixtral-8x7B
The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks we tested.

For full details of this model please read our [release blog post](https://mistral.ai/news/mixtral-of-experts/).

## Warning
This repo contains weights that are compatible with [vLLM](https://github.com/vllm-project/vllm) serving of the model as well as Hugging Face [transformers](https://github.com/huggingface/transformers) library. It is based on the original Mixtral [torrent release](magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce), but the file format and parameter names are different. Please note that model cannot (yet) be instantiated with HF.

## Instruction format

This format must be strictly respected, otherwise the model will generate sub-optimal outputs.

The template used to build a prompt for the Instruct model is defined as follows:
```
<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
```
Note that `<s>` and `</s>` are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings.

As reference, here is the pseudo-code used to tokenize instructions during fine-tuning:
```python
def tokenize(text):
    return tok.encode(text, add_special_tokens=False)

[BOS_ID] + 
tokenize("[INST]") + tokenize(USER_MESSAGE_1) + tokenize("[/INST]") +
tokenize(BOT_MESSAGE_1) + [EOS_ID] +
…
tokenize("[INST]") + tokenize(USER_MESSAGE_N) + tokenize("[/INST]") +
tokenize(BOT_MESSAGE_N) + [EOS_ID]
```

In the pseudo-code above, note that the `tokenize` method should not add a BOS or EOS token automatically, but should add a prefix space. 

## Run the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

By default, transformers will load the model in full precision. Therefore you might be interested to further reduce down the memory requirements to run the model through the optimizations we offer in HF ecosystem:

### In half-precision

Note `float16` precision only works on GPU devices

<details>
<summary> Click to expand </summary>

```diff
+ import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(0)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Lower precision using (8-bit & 4-bit) using `bitsandbytes`

<details>
<summary> Click to expand </summary>

```diff
+ import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Load the model with Flash Attention 2

<details>
<summary> Click to expand </summary>

```diff
+ import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=True)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

## Limitations

The Mixtral-8x7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. 
It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

# The Mistral AI Team
Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Louis Ternon, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.
