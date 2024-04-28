---
language:
- en
license: llama2
tags:
- text generation
- instruct
datasets:
- PygmalionAI/PIPPA
- Open-Orca/OpenOrca
- Norquinal/claude_multiround_chat_30k
- jondurbin/airoboros-gpt4-1.4.1
- databricks/databricks-dolly-15k
model_name: Mythalion 13B
base_model: PygmalionAI/mythalion-13b
inference: false
model_creator: PygmalionAI
model_type: llama
pipeline_tag: text-generation
prompt_template: 'Below is an instruction that describes a task. Write a response
  that appropriately completes the request.


  ### Instruction:

  {prompt}


  ### Response:

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

# Mythalion 13B - AWQ
- Model creator: [PygmalionAI](https://huggingface.co/PygmalionAI)
- Original model: [Mythalion 13B](https://huggingface.co/PygmalionAI/mythalion-13b)

<!-- description start -->
## Description

This repo contains AWQ model files for [PygmalionAI's Mythalion 13B](https://huggingface.co/PygmalionAI/mythalion-13b).


### About AWQ

AWQ is an efficient, accurate and blazing-fast low-bit weight quantization method, currently supporting 4-bit quantization. Compared to GPTQ, it offers faster Transformers-based inference.

It is also now supported by continuous batching server [vLLM](https://github.com/vllm-project/vllm), allowing use of AWQ models for high-throughput concurrent inference in multi-user server scenarios. Note that, at the time of writing, overall throughput is still lower than running vLLM with unquantised models, however using AWQ enables using much smaller GPUs which can lead to easier deployment and overall cost savings. For example, a 70B model can be run on 1 x 48GB GPU instead of 2 x 80GB.
<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Mythalion-13B-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Mythalion-13B-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Mythalion-13B-GGUF)
* [PygmalionAI's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/PygmalionAI/mythalion-13b)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Alpaca

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:

```

<!-- prompt-template end -->


<!-- README_AWQ.md-provided-files start -->
## Provided files and AWQ parameters

For my first release of AWQ models, I am releasing 128g models only. I will consider adding 32g as well if there is interest, and once I have done perplexity and evaluation comparisons, but at this time 32g models are still not fully tested with AutoAWQ and vLLM.

Models are released as sharded safetensors files.

| Branch | Bits | GS | AWQ Dataset | Seq Len | Size |
| ------ | ---- | -- | ----------- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/Mythalion-13B-AWQ/tree/main) | 4 | 128 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 7.25 GB

<!-- README_AWQ.md-provided-files end -->

<!-- README_AWQ.md-use-from-vllm start -->
## Serving this model from vLLM

Documentation on installing and using vLLM [can be found here](https://vllm.readthedocs.io/en/latest/).

- When using vLLM as a server, pass the `--quantization awq` parameter, for example:

```shell
python3 python -m vllm.entrypoints.api_server --model TheBloke/Mythalion-13B-AWQ --quantization awq
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

llm = LLM(model="TheBloke/Mythalion-13B-AWQ", quantization="awq")

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

model_name_or_path = "TheBloke/Mythalion-13B-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

prompt = "Tell me about AI"
prompt_template=f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:

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

# Original model card: PygmalionAI's Mythalion 13B

<h1 style="text-align: center">Mythalion 13B</h1>
<h2 style="text-align: center">A merge of Pygmalion-2 13B and MythoMax 13B</h2>

## Model Details

The long-awaited release of our new models based on Llama-2 is finally here. This model was created in
collaboration with [Gryphe](https://huggingface.co/Gryphe), a mixture of our [Pygmalion-2 13B](https://huggingface.co/PygmalionAI/pygmalion-2-13b)
and Gryphe's [Mythomax L2 13B](https://huggingface.co/Gryphe/MythoMax-L2-13b).

Finer details of the merge are available in [our blogpost](https://pygmalionai.github.io/blog/posts/introducing_pygmalion_2/#mythalion-13b).
According to our testers, this model seems to outperform MythoMax in RP/Chat. **Please make sure you follow the recommended
generation settings for SillyTavern [here](https://pygmalionai.github.io/blog/posts/introducing_pygmalion_2/#sillytavern) for
the best results!**

This model is freely available for both commercial and non-commercial use, as per the Llama-2 license.


## Prompting

This model can be prompted using both the Alpaca and [Pygmalion formatting](https://huggingface.co/PygmalionAI/pygmalion-2-13b#prompting).

**Alpaca formatting**:
```
### Instruction:
<prompt>

### Response:
<leave a newline blank for model to respond>
```

**Pygmalion/Metharme formatting**:
```
<|system|>Enter RP mode. Pretend to be {{char}} whose persona follows:
{{persona}}

You shall reply to the user while staying in character, and generate long responses.
<|user|>Hello!<|model|>{model's response goes here}

```


The model has been trained on prompts using three different roles, which are denoted by the following tokens: `<|system|>`, `<|user|>` and `<|model|>`.

The `<|system|>` prompt can be used to inject out-of-channel information behind the scenes, while the `<|user|>` prompt should be used to indicate user input.
The `<|model|>` token should then be used to indicate that the model should generate a response. These tokens can happen multiple times and be chained up to
form a conversation history.

## Limitations and biases

The intended use-case for this model is fictional writing for entertainment purposes. Any other sort of usage is out of scope.

As such, it was **not** fine-tuned to be safe and harmless: the base model _and_ this fine-tune have been trained on data known to contain profanity and texts that are lewd or otherwise offensive. It may produce socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive. Outputs might often be factually wrong or misleading.

## Acknowledgements
We would like to thank [SpicyChat](https://spicychat.ai/) for sponsoring the training for the [Pygmalion-2 13B](https://huggingface.co/PygmalionAI/pygmalion-2-13b) model.

[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)
