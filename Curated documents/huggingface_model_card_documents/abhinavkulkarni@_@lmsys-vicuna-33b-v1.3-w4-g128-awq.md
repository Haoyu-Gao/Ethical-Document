---
inference: false
---

## Vicuña 33b v1.3 (4-bit 128g AWQ Quantized)

Vicuna is a chat assistant trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT.

- **Developed by:** [LMSYS](https://lmsys.org/)
- **Model type:** An auto-regressive language model based on the transformer architecture.
- **License:** Non-commercial license
- **Finetuned from model:** [LLaMA](https://arxiv.org/abs/2302.13971).

This model is a 4-bit 128 group size AWQ quantized model. For more information about AWQ quantization, please click [here](https://github.com/mit-han-lab/llm-awq).

## Model Date

July 14, 2023

## Model License

Please refer to original Vicuna model license ([link](https://huggingface.co/lmsys/vicuna-33b-v1.3)).

Please refer to the AWQ quantization license ([link](https://github.com/llm-awq/blob/main/LICENSE)).

## CUDA Version

This model was successfully tested on CUDA driver v530.30.02 and runtime v11.7 with Python v3.10.11. Please note that AWQ requires NVIDIA GPUs with compute capability of `8.0` or higher.

For Docker users, the `nvcr.io/nvidia/pytorch:23.06-py3` image is runtime v12.1 but otherwise the same as the configuration above and has also been verified to work.

## How to Use

```bash
git clone https://github.com/mit-han-lab/llm-awq \
&& cd llm-awq \
&& git checkout f084f40bd996f3cf3a0633c1ad7d9d476c318aaa \
&& pip install -e . \
&& cd awq/kernels \
&& python setup.py install
```

```python
import time
import torch
from awq.quantize.quantizer import real_quantize_model_weight
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, TextStreamer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

model_name = "abhinavkulkarni/lmsys-vicuna-33b-v1.3-w4-g128-awq"

# Config
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Model
w_bit = 4
q_config = {
    "zero_point": True,
    "q_group_size": 128,
}

load_quant = snapshot_download(model_name)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config=config, 
                                                 torch_dtype=torch.float16, trust_remote_code=True)

real_quantize_model_weight(model, w_bit=w_bit, q_config=q_config, init_only=True)
model.tie_weights()

model = load_checkpoint_and_dispatch(model, load_quant, device_map="balanced")

# Inference
prompt = f'''What is the difference between nuclear fusion and fission?
###Response:'''

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
output = model.generate(
    inputs=input_ids, 
    temperature=0.7,
    max_new_tokens=512,
    top_p=0.15,
    top_k=0,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    streamer=streamer)
```

## Evaluation

This evaluation was done using [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness).

[vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3)

|  Task  |Version|    Metric     |Value |   |Stderr|
|--------|------:|---------------|-----:|---|------|
|wikitext|      1|word_perplexity|9.8210|   |      |
|        |       |byte_perplexity|1.5330|   |      |
|        |       |bits_per_byte  |0.6163|   |      |

[vicuna-33b-v1.3 (4-bit 128-group AWQ)](https://huggingface.co/abhinavkulkarni/lmsys-vicuna-33b-v1.3-w4-g128-awq)

|  Task  |Version|    Metric     |Value |   |Stderr|
|--------|------:|---------------|-----:|---|------|
|wikitext|      1|word_perplexity|9.9924|   |      |
|        |       |byte_perplexity|1.5380|   |      |
|        |       |bits_per_byte  |0.6210|   |      |

## Acknowledgements

The model was quantized with AWQ technique. If you find AWQ useful or relevant to your research, please kindly cite the paper:

```
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
```
