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

# 🚀 Falcon-180B

**Falcon-180B is a 180B parameters causal decoder-only model built by [TII](https://www.tii.ae) and trained on 3,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. It is made available under the [Falcon-180B TII License](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt) and [Acceptable Use Policy](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/ACCEPTABLE_USE_POLICY.txt).**

*Paper coming soon* 😊


🤗 To get started with Falcon (inference, finetuning, quantization, etc.), we recommend reading [this great blogpost from HF](https://hf.co/blog/falcon-180b) or this [one](https://huggingface.co/blog/falcon) from the release of the 40B!
Note that since the 180B is larger than what can easily be handled with `transformers`+`acccelerate`, we recommend using [Text Generation Inference](https://github.com/huggingface/text-generation-inference).

You will need **at least 400GB of memory** to swiftly run inference with Falcon-180B.

## Why use Falcon-180B?

* **It is the best open-access model currently available, and one of the best model overall.** Falcon-180B outperforms [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-70b-hf), [StableLM](https://github.com/Stability-AI/StableLM), [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1), [MPT](https://huggingface.co/mosaicml/mpt-7b), etc. See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
* **It features an architecture optimized for inference**, with multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)). 
* **It is made available under a permissive license allowing for commercial use**.
* ⚠️ **This is a raw, pretrained model, which should be further finetuned for most usecases.** If you are looking for a version better suited to taking generic instructions in a chat format, we recommend taking a look at [Falcon-180B-Chat](https://huggingface.co/tiiuae/falcon-180b-chat). 


💸 **Looking for a smaller, less expensive model?** [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) and [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) are Falcon-180B's little brothers!

💥 **Falcon LLMs require PyTorch 2.0 for use with `transformers`!**


# Model Card for Falcon-180B

## Model Details

### Model Description

- **Developed by:** [https://www.tii.ae](https://www.tii.ae);
- **Model type:** Causal decoder-only;
- **Language(s) (NLP):** English, German, Spanish, French (and limited capabilities in Italian, Portuguese, Polish, Dutch, Romanian, Czech, Swedish);
- **License:** [Falcon-180B TII License](https://huggingface.co/tiiuae/falcon-180B/blob/main/LICENSE.txt) and [Acceptable Use Policy](https://huggingface.co/tiiuae/falcon-180B/blob/main/ACCEPTABLE_USE_POLICY.txt).

### Model Source

- **Paper:** *coming soon*.

## Uses

See the [acceptable use policy](https://huggingface.co/tiiuae/falcon-180B/blob/main/ACCEPTABLE_USE_POLICY.txt).

### Direct Use

Research on large language models; as a foundation for further specialization and finetuning for specific usecases (e.g., summarization, text generation, chatbot, etc.)

### Out-of-Scope Use

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful. 

## Bias, Risks, and Limitations

Falcon-180B is trained mostly on English, German, Spanish, French, with limited capabilities also in in Italian, Portuguese, Polish, Dutch, Romanian, Czech, Swedish. It will not generalize appropriately to other languages. Furthermore, as it is trained on a large-scale corpora representative of the web, it will carry the stereotypes and biases commonly encountered online.

### Recommendations

We recommend users of Falcon-180B to consider finetuning it for the specific set of tasks of interest, and for guardrails and appropriate precautions to be taken for any production use.

## How to Get Started with the Model

To run inference with the model in full `bfloat16` precision you need approximately 8xA100 80GB or equivalent.



```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-180b"

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

### Training Data

Falcon-180B was trained on 3,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), a high-quality filtered and deduplicated web dataset which we enhanced with curated corpora. Significant components from our curated copora were inspired by The Pile ([Gao et al., 2020](https://arxiv.org/abs/2101.00027)). 

| **Data source**    | **Fraction** | **Tokens** | **Sources**                       |
|--------------------|--------------|------------|-----------------------------------|
| [RefinedWeb-English](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | 75%          | 750B     | massive web crawl                 |
| RefinedWeb-Europe              | 7%           | 70B       | European massive web crawl                                   |
| Books  | 6%           | 60B        |                  |
| Conversations      | 5%           | 50B        | Reddit, StackOverflow, HackerNews |
| Code               | 5%           | 50B        |                                   |
| Technical          | 2%           | 20B        | arXiv, PubMed, USPTO, etc.        |

RefinedWeb-Europe is made of the following languages:

| **Language** | **Fraction of multilingual data** | **Tokens** |
|--------------|-----------------------------------|------------|
| German       | 26%                               | 18B        |
| Spanish      | 24%                               | 17B        |
| French       | 23%                               | 16B        |
| _Italian_    | 7%                                | 5B         |
| _Portuguese_ | 4%                                | 3B         |
| _Polish_     | 4%                                | 3B         |
| _Dutch_      | 4%                                | 3B         |
| _Romanian_   | 3%                                | 2B         |
| _Czech_      | 3%                                | 2B         |
| _Swedish_    | 2%                                | 1B         |


The data was tokenized with the Falcon tokenizer.

### Training Procedure 

Falcon-180B was trained on up to 4,096 A100 40GB GPUs, using a 3D parallelism strategy (TP=8, PP=8, DP=64) combined with ZeRO.

#### Training Hyperparameters

| **Hyperparameter** | **Value**  | **Comment**                               |
|--------------------|------------|-------------------------------------------|
| Precision          | `bfloat16` |                                           |
| Optimizer          | AdamW      |                                           |
| Learning rate      | 1.25e-4       | 4B tokens warm-up, cosine decay to 1.25e-5 |
| Weight decay       | 1e-1       |                                           |
| Z-loss       | 1e-4       |                                           |
| Batch size         | 2048        | 100B tokens ramp-up                         |


#### Speeds, Sizes, Times

Training started in early 2023.

## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


## Technical Specifications 

### Model Architecture and Objective

Falcon-180B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with two layer norms.

For multiquery, we are using an internal variant which uses independent key and values per tensor parallel degree (so-called multigroup).

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 80        |                                        |
| `d_model`          | 14848      |                                        |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-180B was trained on AWS SageMaker, on up to 4,096 A100 40GB GPUs in P4d instances. 

#### Software

Falcon-180B was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)


## Citation

*Paper coming soon* 😊 (actually this time). In the meanwhile, you can use the following information to cite: 
```
@article{falcon,
  title={The Falcon Series of Language Models: Towards Open Frontier Models},
  author={Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Alhammadi, Maitha and Daniele, Mazzotta and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme},
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