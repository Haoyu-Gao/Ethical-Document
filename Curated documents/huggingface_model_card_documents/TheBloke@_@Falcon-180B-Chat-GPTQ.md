---
language:
- en
- de
- es
- fr
license: unknown
datasets:
- tiiuae/falcon-refinedweb
model_name: Falcon 180B Chat
inference: false
model_creator: Technology Innovation Institute
model_link: https://huggingface.co/tiiuae/falcon-180B-chat
model_type: falcon
quantized_by: TheBloke
base_model: tiiuae/falcon-180B-chat
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

# Falcon 180B Chat - GPTQ
- Model creator: [Technology Innovation Institute](https://huggingface.co/tiiuae)
- Original model: [Falcon 180B Chat](https://huggingface.co/tiiuae/falcon-180B-chat)

<!-- description start -->
## Description

This repo contains GPTQ model files for [Technology Innovation Institute's Falcon 180B Chat](https://huggingface.co/tiiuae/falcon-180B-chat).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

## Requirements

Transformers version 4.33.0 is required.

Due to the huge size of the model, the GPTQ has been sharded. This will break compatibility with AutoGPTQ, and therefore any clients/libraries that use AutoGPTQ directly.

But they work great loaded directly through Transformers - and can be served using Text Generation Inference!

## Compatibility

Currently these GPTQs are known to work with:
- Transformers 4.33.0
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) version 1.0.4
  - Docker container: `ghcr.io/huggingface/text-generation-inference:latest` 

<!-- description end -->

<!-- repositories-available start -->
## Repositories available

* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Falcon-180B-Chat-GGUF)
* [Technology Innovation Institute's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/tiiuae/falcon-180B-chat)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Falcon

```
{system_message}
User: {prompt}
Assistant:
```

Example:

```
Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe
User: Hello, Girafatron!
Girafatron:
```
<!-- prompt-template end -->

<!-- README_GPTQ.md-provided-files start -->
## Provided files and GPTQ parameters

Multiple quantisation parameters are provided, to allow you to choose the best one for your hardware and requirements.

Each separate quant is in a different branch.  See below for instructions on fetching from different branches.

All recent GPTQ files are made with AutoGPTQ, and all files in non-main branches are made with AutoGPTQ. Files in the `main` branch which were uploaded before August 2023 were made with GPTQ-for-LLaMa.

<details>
  <summary>Explanation of GPTQ parameters</summary>

- Bits: The bit size of the quantised model.
- GS: GPTQ group size. Higher numbers use less VRAM, but have lower quantisation accuracy. "None" is the lowest possible value.
- Act Order: True or False. Also known as `desc_act`. True results in better quantisation accuracy. Some GPTQ clients have had issues with models that use Act Order plus Group Size, but this is generally resolved now.
- Damp %: A GPTQ parameter that affects how samples are processed for quantisation. 0.01 is default, but 0.1 results in slightly better accuracy.
- GPTQ dataset: The dataset used for quantisation. Using a dataset more appropriate to the model's training can improve quantisation accuracy. Note that the GPTQ dataset is not the same as the dataset used to train the model - please refer to the original model repo for details of the training dataset(s).
- Sequence Length: The length of the dataset sequences used for quantisation. Ideally this is the same as the model sequence length. For some very long sequence models (16+K), a lower sequence length may have to be used.  Note that a lower sequence length does not limit the sequence length of the quantised model. It only impacts the quantisation accuracy on longer inference sequences.
- ExLlama Compatibility: Whether this file can be loaded with ExLlama, which currently only supports Llama models in 4-bit.

</details>

| Branch | Bits | GS | Act Order | Damp % | GPTQ Dataset | Seq Len | Size | ExLlama | Desc |
| ------ | ---- | -- | --------- | ------ | ------------ | ------- | ---- | ------- | ---- |
| main | 4 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 2048 | 94.25 GB | No | 4-bit, with Act Order and group size 128g. Higher quality than group_size=None, but also higher VRAM usage. | 
| gptq-4bit--1g-actorder_True | 4 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 2048 | 92.74 GB | No | 4-bit, with Act Order. No group size, to lower VRAM requirements. |
| gptq-3bit-128g-actorder_True | 3 | 128 | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 2048 | 73.81 GB | No | 3-bit, so much lower VRAM requirements but worse quality than 4-bit. With group size 128g and act-order. Higher quality than 3bit-128g-False. | 
| gptq-3bit--1g-actorder_True | 3 | None | Yes | 0.1 | [wikitext](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test) | 2048 | 70.54 GB | No | 3-bit, so much lower VRAM requirements but worse quality than 4-bit. With no group size for lowest possible VRAM requirements. Lower quality than 3-bit 128g. | 

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download from branches

- In text-generation-webui, you can add `:branch` to the end of the download name, eg `TheBloke/Falcon-180B-Chat-GPTQ:gptq-3bit--1g-actorder_True`
- With Git, you can clone a branch with:
```
git clone --single-branch --branch gptq-3bit--1g-actorder_True https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ
```
- In Python Transformers code, the branch is the `revision` parameter; see below.
<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

**NOTE**: I have not tested this model with Text Generation Webui. It *should* work through the Transformers Loader.  It will *not* work through the AutoGPTQ loader, due to the files being sharded.

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Falcon-180B-Chat-GPTQ`.
  - To download from a specific branch, enter for example `TheBloke/Falcon-180B-Chat-GPTQ:gptq-3bit-128g-actorder_True`
  - see Provided Files above for the list of branches for each option.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. Choose Loader: Transformers
6. In the top left, click the refresh icon next to **Model**.
7. In the **Model** dropdown, choose the model you just downloaded: `Falcon-180B-Chat-GPTQ`
8. The model will automatically load, and is now ready for use!
9. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
10. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!
<!-- README_GPTQ.md-text-generation-webui end -->

<!-- README_GPTQ.md-use-from-python start -->
## How to use this GPTQ model from Python code

### Install the necessary packages

Requires: Transformers 4.33.0 or later, Optimum 1.12.0 or later, and AutoGPTQ.

```shell
pip3 install transformers>=4.33.0 optimum>=1.12.0
pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7
```

### Transformers sample code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Falcon-180B-Chat-GPTQ"

# To use a different branch, change revision
# For example: revision="gptq-3bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''User: {prompt}
Assistant: '''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, do_sample=True, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
```
<!-- README_GPTQ.md-use-from-python end -->

<!-- README_GPTQ.md-compatibility start -->
## Compatibility

The provided files have been tested with Transformers 4.33.0, and TGI 1.0.4.  

Because they are sharded, they will not yet via AutoGPTQ. It is hoped support will be added soon.

Note: lack of support for AutoGPTQ doesn't affect your ability to load these models from Python code. It only affects third-party clients that might use AutoGPTQ.

[Huggingface Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is confirmed working as of version 1.0.4.
<!-- README_GPTQ.md-compatibility end -->

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

**Patreon special mentions**: Russ Johnson, J, alfie_i, Alex, NimbleBox.ai, Chadd, Mandus, Nikolai Manek, Ken Nordquist, ya boyyy, Illia Dulskyi, Viktor Bowallius, vamX, Iucharbius, zynix, Magnesian, Clay Pascal, Pierre Kircher, Enrico Ros, Tony Hughes, Elle, Andrey, knownsqashed, Deep Realms, Jerry Meng, Lone Striker, Derek Yates, Pyrater, Mesiah Bishop, James Bentley, Femi Adebogun, Brandon Frisco, SuperWojo, Alps Aficionado, Michael Dempsey, Vitor Caleffi, Will Dee, Edmond Seymore, usrbinkat, LangChain4j, Kacper Wikieł, Luke Pendergrass, John Detwiler, theTransient, Nathan LeClaire, Tiffany J. Kim, biorpg, Eugene Pentland, Stanislav Ovsiannikov, Fred von Graf, terasurfer, Kalila, Dan Guido, Nitin Borwankar, 阿明, Ai Maven, John Villwock, Gabriel Puliatti, Stephen Murray, Asp the Wyvern, danny, Chris Smitley, ReadyPlayerEmma, S_X, Daniel P. Andersen, Olakabola, Jeffrey Morgan, Imad Khwaja, Caitlyn Gatomon, webtim, Alicia Loh, Trenton Dambrowitz, Swaroop Kallakuri, Erik Bjäreholt, Leonard Tan, Spiking Neurons AB, Luke @flexchar, Ajan Kanaga, Thomas Belote, Deo Leter, RoA, Willem Michiel, transmissions 11, subjectnull, Matthew Berman, Joseph William Delisle, David Ziegler, Michael Davis, Johann-Peter Hartmann, Talal Aujan, senxiiz, Artur Olbinski, Rainer Wilmers, Spencer Kim, Fen Risland, Cap'n Zoog, Rishabh Srivastava, Michael Levine, Geoffrey Montalvo, Sean Connelly, Alexandros Triantafyllidis, Pieter, Gabriel Tamborski, Sam, Subspace Studios, Junyu Yang, Pedro Madruga, Vadim, Cory Kujawski, K, Raven Klaugh, Randy H, Mano Prime, Sebastain Graf, Space Cruiser


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Technology Innovation Institute's Falcon 180B Chat


# 🚀 Falcon-180B-Chat

**Falcon-180B-Chat is a 180B parameters causal decoder-only model built by [TII](https://www.tii.ae) based on [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B) and finetuned on a mixture of [Ultrachat](https://huggingface.co/datasets/stingning/ultrachat), [Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) and [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-2.1). It is made available under the [Falcon-180B TII License](https://huggingface.co/tiiuae/falcon-180B-chat/blob/main/LICENSE.txt) and [Acceptable Use Policy](https://huggingface.co/tiiuae/falcon-180B-chat/blob/main/ACCEPTABLE_USE_POLICY.txt).**

*Paper coming soon* 😊


🤗 To get started with Falcon (inference, finetuning, quantization, etc.), we recommend reading [this great blogpost from HF](https://hf.co/blog/falcon-180b) or this [one](https://huggingface.co/blog/falcon) from the release of the 40B!
Note that since the 180B is larger than what can easily be handled with `transformers`+`acccelerate`, we recommend using [Text Generation Inference](https://github.com/huggingface/text-generation-inference).

You will need **at least 400GB of memory** to swiftly run inference with Falcon-180B.

## Why use Falcon-180B-chat?

* ✨ **You are looking for a ready-to-use chat/instruct model based on [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B).**
* **It is the best open-access model currently available, and one of the best model overall.** Falcon-180B outperforms [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-70b-hf), [StableLM](https://github.com/Stability-AI/StableLM), [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1), [MPT](https://huggingface.co/mosaicml/mpt-7b), etc. See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
* **It features an architecture optimized for inference**, with multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)). 
* **It is made available under a permissive license allowing for commercial use**.

💬 **This is a Chat model, which may not be ideal for further finetuning.** If you are interested in building your own instruct/chat model, we recommend starting from [Falcon-180B](https://huggingface.co/tiiuae/falcon-180b). 

💸 **Looking for a smaller, less expensive model?** [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) and [Falcon-40B-Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) are Falcon-180B-Chat's little brothers!

💥 **Falcon LLMs require PyTorch 2.0 for use with `transformers`!**


# Model Card for Falcon-180B-Chat

## Model Details

### Model Description

- **Developed by:** [https://www.tii.ae](https://www.tii.ae);
- **Model type:** Causal decoder-only;
- **Language(s) (NLP):** English, German, Spanish, French (and limited capabilities in Italian, Portuguese, Polish, Dutch, Romanian, Czech, Swedish);
- **License:** [Falcon-180B TII License](https://huggingface.co/tiiuae/falcon-180B-chat/blob/main/LICENSE.txt) and [Acceptable Use Policy](https://huggingface.co/tiiuae/falcon-180B-chat/blob/main/ACCEPTABLE_USE_POLICY.txt).

### Model Source

- **Paper:** *coming soon*.

## Uses

See the [acceptable use policy](https://huggingface.co/tiiuae/falcon-180B-chat/blob/main/ACCEPTABLE_USE_POLICY.txt).

### Direct Use

Falcon-180B-Chat has been finetuned on a chat dataset.

### Out-of-Scope Use

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful. 

## Bias, Risks, and Limitations

Falcon-180B-Chat is mostly trained on English data, and will not generalize appropriately to other languages. Furthermore, as it is trained on a large-scale corpora representative of the web, it will carry the stereotypes and biases commonly encountered online.

### Recommendations

We recommend users of Falcon-180B-Chat to develop guardrails and to take appropriate precautions for any production use.

## How to Get Started with the Model

To run inference with the model in full `bfloat16` precision you need approximately 8xA100 80GB or equivalent.



```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-180b-chat"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

``` 


## Training Details

**Falcon-180B-Chat is  based on [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B).**

### Training Data
Falcon-180B-Chat is finetuned on a mixture of [Ultrachat](https://huggingface.co/datasets/stingning/ultrachat), [Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) and [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-2.1).

The data was tokenized with the Falcon tokenizer.


## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


## Technical Specifications 

### Model Architecture and Objective

Falcon-180B-Chat is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a two layer norms.

For multiquery, we are using an internal variant which uses independent key and values per tensor parallel degree.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 80        |                                        |
| `d_model`          | 14848      |                                        |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-180B-Chat was trained on AWS SageMaker, on up to 4,096 A100 40GB GPUs in P4d instances. 

#### Software

Falcon-180B-Chat was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)


## Citation

*Paper coming soon* 😊. In the meanwhile, you can use the following information to cite: 
```
@article{falcon,
  title={The Falcon Series of Language Models:Towards Open Frontier Models},
  author={Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme},
  year={2023}
}
```

To learn more about the pretraining dataset, see the 📓 [RefinedWeb paper](https://arxiv.org/abs/2306.01116).

```
@article{refinedweb,
  title={The {R}efined{W}eb dataset for {F}alcon {LLM}: outperforming curated corpora with web data, and web data only},
  author={Guilherme Penedo and Quentin Malartic and Daniel Hesslow and Ruxandra Cojocaru and Alessandro Cappelli and Hamza Alobeidli and Baptiste Pannier and Ebtesam Almazrouei and Julien Launay},
  journal={arXiv preprint arXiv:2306.01116},
  eprint={2306.01116},
  eprinttype = {arXiv},
  url={https://arxiv.org/abs/2306.01116},
  year={2023}
}
```


## Contact
falconllm@tii.ae
