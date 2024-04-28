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

# Llama2 13B Tiefighter - AWQ
- Model creator: [KoboldAI](https://huggingface.co/KoboldAI)
- Original model: [Llama2 13B Tiefighter](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter)

<!-- description start -->
## Description

This repo contains AWQ model files for [KoboldAI's Llama2 13B Tiefighter](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter).


### About AWQ

AWQ is an efficient, accurate and blazing-fast low-bit weight quantization method, currently supporting 4-bit quantization. Compared to GPTQ, it offers faster Transformers-based inference with equivalent or better quality compared to the most commonly used GPTQ settings.

It is supported by:

- [Text Generation Webui](https://github.com/oobabooga/text-generation-webui) - using Loader: AutoAWQ
- [vLLM](https://github.com/vllm-project/vllm) - Llama and Mistral models only
- [Hugging Face Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - for use from Python code

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


<!-- README_AWQ.md-provided-files start -->
## Provided files, and AWQ parameters

For my first release of AWQ models, I am releasing 128g models only. I will consider adding 32g as well if there is interest, and once I have done perplexity and evaluation comparisons, but at this time 32g models are still not fully tested with AutoAWQ and vLLM.

Models are released as sharded safetensors files.

| Branch | Bits | GS | AWQ Dataset | Seq Len | Size |
| ------ | ---- | -- | ----------- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-AWQ/tree/main) | 4 | 128 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 4096 | 7.25 GB

<!-- README_AWQ.md-provided-files end -->

<!-- README_AWQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/LLaMA2-13B-Tiefighter-AWQ`.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `LLaMA2-13B-Tiefighter-AWQ`
7. Select **Loader: AutoAWQ**.
8. Click Load, and the model will load and is now ready for use.
9. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
10. Once you're ready, click the **Text Generation** tab and enter a prompt to get started!
<!-- README_AWQ.md-text-generation-webui end -->

<!-- README_AWQ.md-use-from-vllm start -->
## Multi-user inference server: vLLM

Documentation on installing and using vLLM [can be found here](https://vllm.readthedocs.io/en/latest/).

- Please ensure you are using vLLM version 0.2 or later.
- When using vLLM as a server, pass the `--quantization awq` parameter.

For example:

```shell
python3 python -m vllm.entrypoints.api_server --model TheBloke/LLaMA2-13B-Tiefighter-AWQ --quantization awq
```

- When using vLLM from Python code, again set `quantization=awq`.

For example:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Tell me about AI",
    "Write a story about llamas",
    "What is 291 - 150?",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
]
prompt_template=f'''### Instruction: 
{prompt}
### Response:
'''

prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="TheBloke/LLaMA2-13B-Tiefighter-AWQ", quantization="awq", dtype="auto")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
<!-- README_AWQ.md-use-from-vllm start -->

<!-- README_AWQ.md-use-from-tgi start -->
## Multi-user inference server: Hugging Face Text Generation Inference (TGI)

Use TGI version 1.1.0 or later. The official Docker container is: `ghcr.io/huggingface/text-generation-inference:1.1.0`

Example Docker parameters:

```shell
--model-id TheBloke/LLaMA2-13B-Tiefighter-AWQ --port 3000 --quantize awq --max-input-length 3696 --max-total-tokens 4096 --max-batch-prefill-tokens 4096
```

Example Python code for interfacing with TGI (requires [huggingface-hub](https://github.com/huggingface/huggingface_hub) 0.17.0 or later):

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

print(f"Model output: ", response)
```
<!-- README_AWQ.md-use-from-tgi end -->

<!-- README_AWQ.md-use-from-python start -->
## Inference from Python code using AutoAWQ

### Install the AutoAWQ package

Requires: [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) 0.1.1 or later.

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

### AutoAWQ example code

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/LLaMA2-13B-Tiefighter-AWQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
# Load model
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)

prompt = "Tell me about AI"
prompt_template=f'''### Instruction: 
{prompt}
### Response:
'''

print("*** Running model.generate:")

token_input = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    token_input,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=512
)

# Get the tokens from the output, decode them, print them
token_output = generation_output[0]
text_output = tokenizer.decode(token_output)
print("LLM output: ", text_output)

"""
# Inference should be possible with transformers pipeline as well in future
# But currently this is not yet supported by AutoAWQ (correct as of September 25th 2023)
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
"""
```
<!-- README_AWQ.md-use-from-python end -->

<!-- README_AWQ.md-compatibility start -->
## Compatibility

The files provided are tested to work with:

- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) using `Loader: AutoAWQ`.
- [vLLM](https://github.com/vllm-project/vllm) version 0.2.0 and later.
- [Hugging Face Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) version 1.1.0 and later.
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) version 0.1.1 and later.

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
