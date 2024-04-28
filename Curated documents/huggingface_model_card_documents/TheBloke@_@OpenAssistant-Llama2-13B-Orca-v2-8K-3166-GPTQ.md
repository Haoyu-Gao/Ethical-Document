---
license: other
datasets:
- shahules786/orca-chat
- rombodawg/MegaCodeTraining112k
- theblackcat102/evol-codealpaca-v1
- nickrosh/Evol-Instruct-Code-80k-v1
model_name: Llama2 13B Orca v2 8K
inference: false
model_creator: OpenAssistant
model_link: https://huggingface.co/OpenAssistant/llama2-13b-orca-v2-8k-3166
model_type: llama
quantized_by: TheBloke
base_model: OpenAssistant/llama2-13b-orca-v2-8k-3166
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

# Llama2 13B Orca v2 8K - GPTQ
- Model creator: [OpenAssistant](https://huggingface.co/OpenAssistant)
- Original model: [Llama2 13B Orca v2 8K](https://huggingface.co/OpenAssistant/llama2-13b-orca-v2-8k-3166)

## Description

This repo contains GPTQ model files for [OpenAssistant's Llama2 13B Orca v2 8K](https://huggingface.co/OpenAssistant/llama2-13b-orca-v2-8k-3166).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

## Repositories available

* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGML models for CPU+GPU inference](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GGML)
* [OpenAssistant's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/OpenAssistant/llama2-13b-orca-v2-8k-3166)

## Prompt template: OpenAssistant

```
<|prompter|>{prompt}<|endoftext|><|assistant|>
```

## Provided files and GPTQ parameters

Multiple quantisation parameters are provided, to allow you to choose the best one for your hardware and requirements.

Each separate quant is in a different branch.  See below for instructions on fetching from different branches.

All GPTQ files are made with AutoGPTQ.

<details>
  <summary>Explanation of GPTQ parameters</summary>

- Bits: The bit size of the quantised model.
- GS: GPTQ group size. Higher numbers use less VRAM, but have lower quantisation accuracy. "None" is the lowest possible value.
- Act Order: True or False. Also known as `desc_act`. True results in better quantisation accuracy. Some GPTQ clients have issues with models that use Act Order plus Group Size.
- Damp %: A GPTQ parameter that affects how samples are processed for quantisation. 0.01 is default, but 0.1 results in slightly better accuracy.
- GPTQ dataset: The dataset used for quantisation. The dataset used for quantisation can affect the quantisation accuracy. The dataset used for quantisation is not the same as the dataset used to train the model.
- Sequence Length: The length of the dataset sequences used for quantisation. Ideally this is the same as the model sequence length. For some very long sequence models (16+K), a lower sequence length may have to be used.  Note that a lower sequence length does not limit the sequence length of the quantised model. It only affects the quantisation accuracy on longer inference sequences.
- ExLlama Compatibility: Whether this file can be loaded with ExLlama, which currently only supports Llama models in 4-bit.

</details>

| Branch | Bits | GS | Act Order | Damp % | GPTQ Dataset | Seq Len | Size | ExLlama | Desc |
| ------ | ---- | -- | --------- | ------ | ------------ | ------- | ---- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ/tree/main) | 4 | 128 | No | 0.1 | [Evol Instruct Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 8192 | 7.26 GB | Yes | Most compatible option. Good inference speed in AutoGPTQ and GPTQ-for-LLaMa. Lower inference quality than other options. |
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [Evol Instruct Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 8192 | 8.00 GB | Yes | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. Poor AutoGPTQ CUDA speed. |
| [gptq-4bit-64g-actorder_True](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ/tree/gptq-4bit-64g-actorder_True) | 4 | 64 | Yes | 0.1 | [Evol Instruct Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 8192 | 7.51 GB | Yes | 4-bit, with Act Order and group size 64g. Uses less VRAM than 32g, but with slightly lower accuracy. Poor AutoGPTQ CUDA speed. |
| [gptq-4bit-128g-actorder_True](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ/tree/gptq-4bit-128g-actorder_True) | 4 | 128 | Yes | 0.1 | [Evol Instruct Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 8192 | 7.26 GB | Yes | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. Poor AutoGPTQ CUDA speed. |
| [gptq-8bit--1g-actorder_True](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ/tree/gptq-8bit--1g-actorder_True) | 8 | None | Yes | 0.1 | [Evol Instruct Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 8192 | 13.36 GB | No | 8-bit, with Act Order. No group size, to lower VRAM requirements and to improve AutoGPTQ speed. |
| [gptq-8bit-128g-actorder_True](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ/tree/gptq-8bit-128g-actorder_True) | 8 | 128 | Yes | 0.1 | [Evol Instruct Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 8192 | 13.65 GB | No | 8-bit, with group size 128g for higher inference quality and with Act Order for even higher accuracy. Poor AutoGPTQ CUDA speed. |

## How to download from branches

- In text-generation-webui, you can add `:branch` to the end of the download name, eg `TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ:gptq-4bit-32g-actorder_True`
- With Git, you can clone a branch with:
```
git clone --single-branch --branch gptq-4bit-32g-actorder_True https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ
```
- In Python Transformers code, the branch is the `revision` parameter; see below.

## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ`.
  - To download from a specific branch, enter for example `TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ:gptq-4bit-32g-actorder_True`
  - see Provided Files above for the list of branches for each option.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done"
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to set GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!

## How to use this GPTQ model from Python code

First make sure you have [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 0.3.1 or later installed:

```
pip3 install auto-gptq
```

If you have problems installing AutoGPTQ, please build from source instead:
```
pip3 uninstall -y auto-gptq
git clone https://github.com/PanQiWei/AutoGPTQ
cd AutoGPTQ
pip3 install .
```

Then try the following example code:

```python
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/OpenAssistant-Llama2-13B-Orca-v2-8K-3166-GPTQ"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

"""
# To download from a specific branch, use the revision parameter, as in this example:
# Note that `revision` requires AutoGPTQ 0.3.1 or later!

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision="gptq-4bit-32g-actorder_True",
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        quantize_config=None)
"""

prompt = "Tell me about AI"
prompt_template=f'''<|prompter|>{prompt}<|endoftext|><|assistant|>
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
```

## Compatibility

The files provided will work with AutoGPTQ (CUDA and Triton modes), GPTQ-for-LLaMa (only CUDA has been tested), and Occ4m's GPTQ-for-LLaMa fork.

ExLlama works with Llama models in 4-bit. Please see the Provided Files table above for per-file compatibility.

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute.

Thanks to the [chirper.ai](https://chirper.ai) team!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Sam, theTransient, Jonathan Leane, Steven Wood, webtim, Johann-Peter Hartmann, Geoffrey Montalvo, Gabriel Tamborski, Willem Michiel, John Villwock, Derek Yates, Mesiah Bishop, Eugene Pentland, Pieter, Chadd, Stephen Murray, Daniel P. Andersen, terasurfer, Brandon Frisco, Thomas Belote, Sid, Nathan LeClaire, Magnesian, Alps Aficionado, Stanislav Ovsiannikov, Alex, Joseph William Delisle, Nikolai Manek, Michael Davis, Junyu Yang, K, J, Spencer Kim, Stefan Sabev, Olusegun Samson, transmissions 11, Michael Levine, Cory Kujawski, Rainer Wilmers, zynix, Kalila, Luke @flexchar, Ajan Kanaga, Mandus, vamX, Ai Maven, Mano Prime, Matthew Berman, subjectnull, Vitor Caleffi, Clay Pascal, biorpg, alfie_i, 阿明, Jeffrey Morgan, ya boyyy, Raymond Fosdick, knownsqashed, Olakabola, Leonard Tan, ReadyPlayerEmma, Enrico Ros, Dave, Talal Aujan, Illia Dulskyi, Sean Connelly, senxiiz, Artur Olbinski, Elle, Raven Klaugh, Fen Risland, Deep Realms, Imad Khwaja, Fred von Graf, Will Dee, usrbinkat, SuperWojo, Alexandros Triantafyllidis, Swaroop Kallakuri, Dan Guido, John Detwiler, Pedro Madruga, Iucharbius, Viktor Bowallius, Asp the Wyvern, Edmond Seymore, Trenton Dambrowitz, Space Cruiser, Spiking Neurons AB, Pyrater, LangChain4j, Tony Hughes, Kacper Wikieł, Rishabh Srivastava, David Ziegler, Luke Pendergrass, Andrey, Gabriel Puliatti, Lone Striker, Sebastain Graf, Pierre Kircher, Randy H, NimbleBox.ai, Vadim, danny, Deo Leter


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: OpenAssistant's Llama2 13B Orca v2 8K

- wandb: [jlhr5cf2](https://wandb.ai/open-assistant/supervised-finetuning/runs/jlhr5cf2)
- sampling-report: [2023-07-31_OpenAssistant_llama2-13b-orca-v2-8k-3166_sampling_llama2_prompt.json](https://open-assistant.github.io/oasst-model-eval/?f=https%3A%2F%2Fraw.githubusercontent.com%2FOpen-Assistant%2Foasst-model-eval%2Fmain%2Fsampling_reports%2Foasst-pretrained%2F2023-07-31_OpenAssistant_llama2-13b-orca-v2-8k-3166_sampling_llama2_prompt.json)


## Model Configuration
```
llama2-13b-orca-v2-8k:
  rng_seed: 0xe1291f21
  show_dataset_stats: true
  random_offset_probability: 0.0
  use_custom_sampler: true
  sort_by_length: false
  dtype: fp16
  log_dir: /mnt/data/ikka/data_cache/llama2_13b_orcav2_logs
  output_dir: /mnt/data/ikka/data_cache/llama2_13b_orcav2
  learning_rate: 1e-5
  model_name: conceptofmind/LLongMA-2-13b
  deepspeed_config: configs/zero_config_pretrain.json
  weight_decay: 0.000001
  max_length: 8192
  warmup_steps: 100
  peft_model: false
  use_flash_attention: true
  gradient_checkpointing: true
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 1
  residual_dropout: 0.0
  eval_steps: 200
  save_steps: 200
  num_train_epochs: 1
  save_total_limit: 4
  superhot: false
  superhot_config:
    type: linear
    scaling_factor: 2
  datasets:
    - orca-chat: # shahules786/orca-chat
        data_files: orca-chat-gpt4-8k.json
        max_val_set: 5000
        val_split: 0.1
    - evol-codealpaca-v1: # theblackcat102/evol-codealpaca-v1
        fill_min_length: 20000
        val_split: 0.1
    - megacode: # rombodawg/MegaCodeTraining112k
        fill_min_length: 24000
        val_split: 0.1
        max_val_set: 1000
    - evol_instruct_code: # nickrosh/Evol-Instruct-Code-80k-v1
        fill_min_length: 24000
        val_split: 0.1
        max_val_set: 1000
```
