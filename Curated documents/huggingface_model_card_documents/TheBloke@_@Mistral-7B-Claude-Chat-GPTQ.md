---
license: cc-by-nc-4.0
datasets:
- Norquinal/claude_multiround_chat_1k
model_name: Mistral 7B Claude Chat
base_model: Norquinal/Mistral-7B-claude-chat
inference: false
model_creator: Andy B. Norquinal
model_type: mistral
prompt_template: 'A chat between a curious user and an artificial intelligence assistant.
  The assistant gives helpful, detailed, and polite answers to the user''s questions.
  USER: {prompt} ASSISTANT:

  '
quantized_by: TheBloke
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

# Mistral 7B Claude Chat - GPTQ
- Model creator: [Andy B. Norquinal](https://huggingface.co/Norquinal)
- Original model: [Mistral 7B Claude Chat](https://huggingface.co/Norquinal/Mistral-7B-claude-chat)

<!-- description start -->
## Description

This repo contains GPTQ model files for [Andy B. Norquinal's Mistral 7B Claude Chat](https://huggingface.co/Norquinal/Mistral-7B-claude-chat).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

These files were quantised using hardware kindly provided by [Massed Compute](https://massedcompute.com/).

<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GGUF)
* [Andy B. Norquinal's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/Norquinal/Mistral-7B-claude-chat)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Vicuna

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:

```

<!-- prompt-template end -->



<!-- README_GPTQ.md-compatible clients start -->
## Known compatible clients / servers

These GPTQ models are known to work in the following inference servers/webuis.

- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [KobaldAI United](https://github.com/henk717/koboldai)
- [LoLLMS Web UI](https://github.com/ParisNeo/lollms-webui)
- [Hugging Face Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)

This may not be a complete list; if you know of others, please let me know!
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
| [main](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ/tree/main) | 4 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 4.16 GB | Yes | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. | 
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 4.57 GB | Yes | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. | 
| [gptq-8bit--1g-actorder_True](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ/tree/gptq-8bit--1g-actorder_True) | 8 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 7.52 GB | No | 8-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-8bit-128g-actorder_True](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ/tree/gptq-8bit-128g-actorder_True) | 8 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 7.68 GB | No | 8-bit, with group size 128g for higher inference quality and with Act Order for even higher accuracy. | 
| [gptq-8bit-32g-actorder_True](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ/tree/gptq-8bit-32g-actorder_True) | 8 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 8.17 GB | No | 8-bit, with group size 32g and Act Order for maximum inference quality. | 
| [gptq-4bit-64g-actorder_True](https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ/tree/gptq-4bit-64g-actorder_True) | 4 | 64 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 4.29 GB | Yes | 4-bit, with Act Order and group size 64g. Uses less VRAM than 32g, but with slightly lower accuracy. |

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download, including from branches

### In text-generation-webui

To download from the `main` branch, enter `TheBloke/Mistral-7B-Claude-Chat-GPTQ` in the "Download model" box.

To download from another branch, add `:branchname` to the end of the download name, eg `TheBloke/Mistral-7B-Claude-Chat-GPTQ:gptq-4bit-32g-actorder_True`

### From the command line

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

To download the `main` branch to a folder called `Mistral-7B-Claude-Chat-GPTQ`:

```shell
mkdir Mistral-7B-Claude-Chat-GPTQ
huggingface-cli download TheBloke/Mistral-7B-Claude-Chat-GPTQ --local-dir Mistral-7B-Claude-Chat-GPTQ --local-dir-use-symlinks False
```

To download from a different branch, add the `--revision` parameter:

```shell
mkdir Mistral-7B-Claude-Chat-GPTQ
huggingface-cli download TheBloke/Mistral-7B-Claude-Chat-GPTQ --revision gptq-4bit-32g-actorder_True --local-dir Mistral-7B-Claude-Chat-GPTQ --local-dir-use-symlinks False
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
mkdir Mistral-7B-Claude-Chat-GPTQ
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/Mistral-7B-Claude-Chat-GPTQ --local-dir Mistral-7B-Claude-Chat-GPTQ --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>

### With `git` (**not** recommended)

To clone a specific branch with `git`, use a command like this:

```shell
git clone --single-branch --branch gptq-4bit-32g-actorder_True https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GPTQ
```

Note that using Git with HF repos is strongly discouraged. It will be much slower than using `huggingface-hub`, and will use twice as much disk space as it has to store the model files twice (it stores every byte both in the intended target folder, and again in the `.git` folder as a blob.)

<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Mistral-7B-Claude-Chat-GPTQ`.

    - To download from a specific branch, enter for example `TheBloke/Mistral-7B-Claude-Chat-GPTQ:gptq-4bit-32g-actorder_True`
    - see Provided Files above for the list of branches for each option.

3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `Mistral-7B-Claude-Chat-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.

    - Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.

9. Once you're ready, click the **Text Generation** tab and enter a prompt to get started!

<!-- README_GPTQ.md-text-generation-webui end -->

<!-- README_GPTQ.md-use-from-tgi start -->
## Serving this model from Text Generation Inference (TGI)

It's recommended to use TGI version 1.1.0 or later. The official Docker container is: `ghcr.io/huggingface/text-generation-inference:1.1.0`

Example Docker parameters:

```shell
--model-id TheBloke/Mistral-7B-Claude-Chat-GPTQ --port 3000 --quantize gptq --max-input-length 3696 --max-total-tokens 4096 --max-batch-prefill-tokens 4096
```

Example Python code for interfacing with TGI (requires huggingface-hub 0.17.0 or later):

```shell
pip3 install huggingface-hub
```

```python
from huggingface_hub import InferenceClient

endpoint_url = "https://your-endpoint-url-here"

prompt = "Tell me about AI"
prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:
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

model_name_or_path = "TheBloke/Mistral-7B-Claude-Chat-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
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

The files provided are tested to work with Transformers. For non-Mistral models, AutoGPTQ can also be used directly.

[ExLlama](https://github.com/turboderp/exllama) is compatible with Llama and Mistral models in 4-bit. Please see the Provided Files table above for per-file compatibility.

For a list of clients/servers, please see "Known compatible clients / servers", above.
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

**Patreon special mentions**: Brandon Frisco, LangChain4j, Spiking Neurons AB, transmissions 11, Joseph William Delisle, Nitin Borwankar, Willem Michiel, Michael Dempsey, vamX, Jeffrey Morgan, zynix, jjj, Omer Bin Jawed, Sean Connelly, jinyuan sun, Jeromy Smith, Shadi, Pawan Osman, Chadd, Elijah Stavena, Illia Dulskyi, Sebastain Graf, Stephen Murray, terasurfer, Edmond Seymore, Celu Ramasamy, Mandus, Alex, biorpg, Ajan Kanaga, Clay Pascal, Raven Klaugh, 阿明, K, ya boyyy, usrbinkat, Alicia Loh, John Villwock, ReadyPlayerEmma, Chris Smitley, Cap'n Zoog, fincy, GodLy, S_X, sidney chen, Cory Kujawski, OG, Mano Prime, AzureBlack, Pieter, Kalila, Spencer Kim, Tom X Nguyen, Stanislav Ovsiannikov, Michael Levine, Andrey, Trailburnt, Vadim, Enrico Ros, Talal Aujan, Brandon Phillips, Jack West, Eugene Pentland, Michael Davis, Will Dee, webtim, Jonathan Leane, Alps Aficionado, Rooh Singh, Tiffany J. Kim, theTransient, Luke @flexchar, Elle, Caitlyn Gatomon, Ari Malik, subjectnull, Johann-Peter Hartmann, Trenton Dambrowitz, Imad Khwaja, Asp the Wyvern, Emad Mostaque, Rainer Wilmers, Alexandros Triantafyllidis, Nicholas, Pedro Madruga, SuperWojo, Harry Royden McLaughlin, James Bentley, Olakabola, David Ziegler, Ai Maven, Jeff Scroggin, Nikolai Manek, Deo Leter, Matthew Berman, Fen Risland, Ken Nordquist, Manuel Alberto Morcote, Luke Pendergrass, TL, Fred von Graf, Randy H, Dan Guido, NimbleBox.ai, Vitor Caleffi, Gabriel Tamborski, knownsqashed, Lone Striker, Erik Bjäreholt, John Detwiler, Leonard Tan, Iucharbius


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Andy B. Norquinal's Mistral 7B Claude Chat

This is the [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) model fine-tuned using QLoRA (4-bit precision) on my [claude_multiround_chat_1k](https://huggingface.co/datasets/Norquinal/claude_multiround_chat_1k) dataset, which is a randomized subset of ~1000 samples from my [claude_multiround_chat_30k](https://huggingface.co/datasets/Norquinal/claude_multiround_chat_30k) dataset.

## Prompt Format
The model was finetuned with the Vicuna 1.1 prompt format:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}
ASSISTANT:
```

## Use in Text Generation Web UI
Install the bleeding-edge version of `transformers` from source:
```
pip install git+https://github.com/huggingface/transformers
```
Or, alternatively, change `model_type` in `config.json` from `mistral` to `llama`.

## Example Usage
>USER: First, give a brief overview of Chinese cuisine, including its regional diversity. next, describe the importance of the five flavors in Chinese cooking. Then, mention three popular Chinese dishes and their main ingredients. After that, provide a popular Chinese cooking technique and an example of a dish prepared using that technique. Finally, discuss the role of tea in traditional Chinese culture.

>ASSISTANT: Chinese cuisine is incredibly diverse due to its large geographic area with different climates and resources available. Regional specialties include Sichuan spicy hot pot, Cantonese dim sum and seafood dishes, Beijing roast duck, and Hunan stir-fries.   
>
>The five basic tastes that guide Chinese cookery are sweetness, sourness, bitterness, saltiness, and pungency (spiciness). These contrasting flavors are used together to create balance in each dish.    
>
>Some classic Chinese dishes include dumplings filled with meat or vegetables, spring rolls stuffed with vegetables and meat wrapped in thin pastry, and noodles served in soups or stir fried with sauces and vegetables.    
>
>A common Chinese cooking technique is braising, where food is slowly cooked in a small amount of liquid until tender and flavorful. An example would be beef brisket braised in soy sauce, ginger and star anise for hours until falling apart.  
>
>Tea plays a central role in Chinese culture as it represents hospitality, relaxation and socializing. Green teas are most commonly consumed but black, oolong and white varieties also exist. Tea drinking rituals involve preparing and serving tea in specific ways depending on context and occasion. 
>
>In summary, Chinese cuisine offers a rich culinary tradition influenced by regional differences, an emphasis on balancing the five basic flavors, and a focus on simple yet flavorful preparations using fresh seasonal produce and high quality meats. Tea acts as both a refreshment and symbol of cultural refinement within Chinese society. I hope this overview provides an interesting starting point for further exploration into the fascinating world of Chinese gastronomy!
