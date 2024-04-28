---
language:
- en
- de
- es
- fr
license: unknown
datasets:
- tiiuae/falcon-refinedweb
inference: false
extra_gated_heading: Acknowledge license to access the repository
extra_gated_prompt: You agree to the [Falcon-180B TII license](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt)
  and [acceptable use policy](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/ACCEPTABLE_USE_POLICY.txt).
extra_gated_button_content: I agree to the terms and conditions of the Falcon-180B
  TII license and to the acceptable use policy
---

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