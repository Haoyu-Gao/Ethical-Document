---
license: llama2
model_name: Llama2 13B Tiefighter
base_model: KoboldAI/LLaMA2-13B-Tiefighter
inference: false
model_creator: KoboldAI
model_type: llama
prompt_template: "### Instruction: \n{prompt}\n### Response:\n"
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

# Llama2 13B Tiefighter - GPTQ
- Model creator: [KoboldAI](https://huggingface.co/KoboldAI)
- Original model: [Llama2 13B Tiefighter](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter)

<!-- description start -->
## Description

This repo contains GPTQ model files for [KoboldAI's Llama2 13B Tiefighter](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GGUF)
* [KoboldAI's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Alpaca-Tiefighter

```
### Instruction: 
{prompt}
### Response:

```

<!-- prompt-template end -->



<!-- README_GPTQ.md-compatible clients start -->
## Known compatible clients / servers

These GPTQs are known to work in the following inference servers/webuis:

- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [KobaldAI United](https://github.com/henk717/koboldai)
- [LoLLMS Web UI](https://github.com/ParisNeo/lollms-webui)
- [Hugging Face Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
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
- ExLlama Compatibility: Whether this file can be loaded with ExLlama, which currently only supports Llama models in 4-bit.

</details>

| Branch | Bits | GS | Act Order | Damp % | GPTQ Dataset | Seq Len | Size | ExLlama | Desc |
| ------ | ---- | -- | --------- | ------ | ------------ | ------- | ---- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ/tree/main) | 4 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 7.26 GB | Yes | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. | 
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 8.00 GB | Yes | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. | 
| [gptq-8bit--1g-actorder_True](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ/tree/gptq-8bit--1g-actorder_True) | 8 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 13.36 GB | No | 8-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-8bit-128g-actorder_True](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ/tree/gptq-8bit-128g-actorder_True) | 8 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 13.65 GB | No | 8-bit, with group size 128g for higher inference quality and with Act Order for even higher accuracy. | 
| [gptq-8bit-32g-actorder_True](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ/tree/gptq-8bit-32g-actorder_True) | 8 | 32 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 14.54 GB | No | 8-bit, with group size 32g and Act Order for maximum inference quality. | 
| [gptq-4bit-64g-actorder_True](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ/tree/gptq-4bit-64g-actorder_True) | 4 | 64 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 7.51 GB | Yes | 4-bit, with Act Order and group size 64g. Uses less VRAM than 32g, but with slightly lower accuracy. |

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download, including from branches

### In text-generation-webui

To download from the `main` branch, enter `TheBloke/LLaMA2-13B-Tiefighter-GPTQ` in the "Download model" box.

To download from another branch, add `:branchname` to the end of the download name, eg `TheBloke/LLaMA2-13B-Tiefighter-GPTQ:gptq-4bit-32g-actorder_True`

### From the command line

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

To download the `main` branch to a folder called `LLaMA2-13B-Tiefighter-GPTQ`:

```shell
mkdir LLaMA2-13B-Tiefighter-GPTQ
huggingface-cli download TheBloke/LLaMA2-13B-Tiefighter-GPTQ --local-dir LLaMA2-13B-Tiefighter-GPTQ --local-dir-use-symlinks False
```

To download from a different branch, add the `--revision` parameter:

```shell
mkdir LLaMA2-13B-Tiefighter-GPTQ
huggingface-cli download TheBloke/LLaMA2-13B-Tiefighter-GPTQ --revision gptq-4bit-32g-actorder_True --local-dir LLaMA2-13B-Tiefighter-GPTQ --local-dir-use-symlinks False
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
mkdir LLaMA2-13B-Tiefighter-GPTQ
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/LLaMA2-13B-Tiefighter-GPTQ --local-dir LLaMA2-13B-Tiefighter-GPTQ --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>

### With `git` (**not** recommended)

To clone a specific branch with `git`, use a command like this:

```shell
git clone --single-branch --branch gptq-4bit-32g-actorder_True https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GPTQ
```

Note that using Git with HF repos is strongly discouraged. It will be much slower than using `huggingface-hub`, and will use twice as much disk space as it has to store the model files twice (it stores every byte both in the intended target folder, and again in the `.git` folder as a blob.)

<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/LLaMA2-13B-Tiefighter-GPTQ`.

    - To download from a specific branch, enter for example `TheBloke/LLaMA2-13B-Tiefighter-GPTQ:gptq-4bit-32g-actorder_True`
    - see Provided Files above for the list of branches for each option.

3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `LLaMA2-13B-Tiefighter-GPTQ`
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
--model-id TheBloke/LLaMA2-13B-Tiefighter-GPTQ --port 3000 --quantize gptq --max-input-length 3696 --max-total-tokens 4096 --max-batch-prefill-tokens 4096
```

Example Python code for interfacing with TGI (requires huggingface-hub 0.17.0 or later):

```shell
pip3 install huggingface-hub
```

```python
from huggingface_hub import InferenceClient

endpoint_url = "https://your-endpoint-url-here"

prompt = "Tell me about AI"
prompt_template=f'''### Instruction: 
{prompt}
### Response:
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

model_name_or_path = "TheBloke/LLaMA2-13B-Tiefighter-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''### Instruction: 
{prompt}
### Response:
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

**Patreon special mentions**: Pierre Kircher, Stanislav Ovsiannikov, Michael Levine, Eugene Pentland, Andrey, 준교 김, Randy H, Fred von Graf, Artur Olbinski, Caitlyn Gatomon, terasurfer, Jeff Scroggin, James Bentley, Vadim, Gabriel Puliatti, Harry Royden McLaughlin, Sean Connelly, Dan Guido, Edmond Seymore, Alicia Loh, subjectnull, AzureBlack, Manuel Alberto Morcote, Thomas Belote, Lone Striker, Chris Smitley, Vitor Caleffi, Johann-Peter Hartmann, Clay Pascal, biorpg, Brandon Frisco, sidney chen, transmissions 11, Pedro Madruga, jinyuan sun, Ajan Kanaga, Emad Mostaque, Trenton Dambrowitz, Jonathan Leane, Iucharbius, usrbinkat, vamX, George Stoitzev, Luke Pendergrass, theTransient, Olakabola, Swaroop Kallakuri, Cap'n Zoog, Brandon Phillips, Michael Dempsey, Nikolai Manek, danny, Matthew Berman, Gabriel Tamborski, alfie_i, Raymond Fosdick, Tom X Nguyen, Raven Klaugh, LangChain4j, Magnesian, Illia Dulskyi, David Ziegler, Mano Prime, Luis Javier Navarrete Lozano, Erik Bjäreholt, 阿明, Nathan Dryer, Alex, Rainer Wilmers, zynix, TL, Joseph William Delisle, John Villwock, Nathan LeClaire, Willem Michiel, Joguhyik, GodLy, OG, Alps Aficionado, Jeffrey Morgan, ReadyPlayerEmma, Tiffany J. Kim, Sebastain Graf, Spencer Kim, Michael Davis, webtim, Talal Aujan, knownsqashed, John Detwiler, Imad Khwaja, Deo Leter, Jerry Meng, Elijah Stavena, Rooh Singh, Pieter, SuperWojo, Alexandros Triantafyllidis, Stephen Murray, Ai Maven, ya boyyy, Enrico Ros, Ken Nordquist, Deep Realms, Nicholas, Spiking Neurons AB, Elle, Will Dee, Jack West, RoA, Luke @flexchar, Viktor Bowallius, Derek Yates, Subspace Studios, jjj, Toran Billups, Asp the Wyvern, Fen Risland, Ilya, NimbleBox.ai, Chadd, Nitin Borwankar, Emre, Mandus, Leonard Tan, Kalila, K, Trailburnt, S_X, Cory Kujawski


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: KoboldAI's Llama2 13B Tiefighter

# LLaMA2-13B-Tiefighter
Tiefighter is a merged model achieved trough merging two different lora's on top of a well established existing merge.
To achieve this the following recipe was used:

* We begin with the base model Undi95/Xwin-MLewd-13B-V0.2 which is a well established merged, contrary to the name this model does not have a strong NSFW bias.
* Then we applied the PocketDoc/Dans-RetroRodeo-13b lora which is a finetune on the Choose your own Adventure datasets from our Skein model.
* After applying this lora we merged the new model with PocketDoc/Dans-RetroRodeo-13b at 5% to weaken the newly introduced adventure bias.
* The resulting merge was used as a new basemodel to which we applied Blackroot/Llama-2-13B-Storywriter-LORA and repeated the same trick, this time at 10%.

This means this model contains the following ingredients from their upstream models for as far as we can track them:
- Undi95/Xwin-MLewd-13B-V0.2
- - Undi95/ReMM-S-Light
  - Undi95/CreativeEngine
  - Brouz/Slerpeno
  - - elinas/chronos-13b-v2
    - jondurbin/airoboros-l2-13b-2.1
    - NousResearch/Nous-Hermes-Llama2-13b+nRuaif/Kimiko-v2
    - CalderaAI/13B-Legerdemain-L2+lemonilia/limarp-llama2-v2
    - - KoboldAI/LLAMA2-13B-Holodeck-1
      - NousResearch/Nous-Hermes-13b
      - OpenAssistant/llama2-13b-orca-8k-3319
    - ehartford/WizardLM-1.0-Uncensored-Llama2-13b
    - Henk717/spring-dragon
  - The-Face-Of-Goonery/Huginn-v3-13b (Contains undisclosed model versions, those we assumed where possible)
  - - SuperCOT (Undisclosed version)
    - elinas/chronos-13b-v2 (Version assumed)
    - NousResearch/Nous-Hermes-Llama2-13b
    - stabilityai/StableBeluga-13B (Version assumed)
  - zattio770/120-Days-of-LORA-v2-13B
  - PygmalionAI/pygmalion-2-13b
  - Undi95/Storytelling-v1-13B-lora
  - TokenBender/sakhi_13B_roleplayer_NSFW_chat_adapter
  - nRuaif/Kimiko-v2-13B
  - The-Face-Of-Goonery/Huginn-13b-FP16
  - - "a lot of different models, like hermes, beluga, airoboros, chronos.. limarp"
  - lemonilia/LimaRP-Llama2-13B-v3-EXPERIMENT
  - Xwin-LM/Xwin-LM-13B-V0.2
- PocketDoc/Dans-RetroRodeo-13b
- Blackroot/Llama-2-13B-Storywriter-LORA

While we could possibly not credit every single lora or model involved in this merged model, we'd like to thank all involved creators upstream for making this awesome model possible!
Thanks to you the AI ecosystem is thriving, and without your dedicated tuning efforts models such as this one would not be possible.

# Usage
This model is meant to be creative, If you let it improvise you get better results than if you drown it in details.

## Story Writing
Regular story writing in the traditional way is supported, simply copy paste your story and continue writing. Optionally use an instruction in memory or an authors note to guide the direction of your story.

### Generate a story on demand
To generate stories on demand you can use an instruction (tested in the Alpaca format) such as "Write a novel about X, use chapters and dialogue" this will generate a story. The format can vary between generations depending on how the model chooses to begin, either write what you want as shown in the earlier example or write the beginning of the story yourself so the model can follow your style. A few retries can also help if the model gets it wrong.

## Chatbots and persona's
This model has been tested with various forms of chatting, testers have found that typically less is more and the model is good at improvising. Don't drown the model in paragraphs of detailed information, instead keep it simple first and see how far you can lean on the models own ability to figure out your character. Copy pasting paragraphs of background information is not suitable for a 13B model such as this one, code formatted characters or an instruction prompt describing who you wish to talk to goes much further.

For example, you can put this in memory in regular chat mode: 
``` 
### Instruction: 
Generate a conversation between Alice and Henk where they discuss language models.
In this conversation Henk is excited to teach Alice about Tiefigther. 
### Response: 
```

Because the model is a merge of a variety of models, it should support a broad range of instruct formats, or plain chat mode. If you have a particular favourite try it, otherwise we recommend to either use the regular chat mode or Alpaca's format.

## Instruct Prompting
This model features various instruct models on a variety of instruction styles, when testing the model we have used Alpaca for our own tests. If you prefer a different format chances are it can work.

During instructions we have observed that in some cases the adventure data can leak, it may also be worth experimenting using > as the prefix for a user command to remedy this. But this may result in a stronger fiction bias.

Keep in mind that while this model can be used as a factual instruct model, the focus was on fiction. Information provided by the model can be made up.

## Adventuring and Adventure Games
This model contains a lora that was trained on the same adventure dataset as the KoboldAI Skein model. Adventuring is best done using an small introduction to the world and your objective while using the > prefix for a user command (KoboldAI's adventure mode).

It is possible that the model does not immediately pick up on what you wish to do and does not engage in its Adventure mode behaviour right away. Simply manually correct the output to trim excess dialogue or other undesirable behaviour and continue to submit your actions using the appropriate mode. The model should pick up on this style quickly and will correctly follow this format within 3 turns.

## Discovered something cool and want to engage with us? 
Join our community at https://koboldai.org/discord !
