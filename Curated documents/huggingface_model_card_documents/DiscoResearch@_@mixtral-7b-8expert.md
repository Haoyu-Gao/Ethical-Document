---
language:
- en
- fr
- it
- es
- de
license: apache-2.0
---

# Mixtral 7b 8 Expert

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62e3b6ab0c2a907c388e4965/6m3e2d2BNXDjy6_qHd2LT.png)

This is a preliminary HuggingFace implementation of the newly released MoE model by MistralAi. Make sure to  load with `trust_remote_code=True`.

Thanks to @dzhulgakov for his early implementation (https://github.com/dzhulgakov/llama-mistral) that helped me find a working setup.

Also many thanks to our friends at [LAION](https://laion.ai) and [HessianAI](https://hessian.ai/) for the compute used for these projects!

Benchmark scores:
```
hella swag: 0.8661
winogrande: 0.824
truthfulqa_mc2: 0.4855
arc_challenge:  0.6638
gsm8k: 0.5709
MMLU: 0.7173
```

# Basic Inference setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("DiscoResearch/mixtral-7b-8expert", low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("DiscoResearch/mixtral-7b-8expert")
x = tok.encode("The mistral wind in is a phenomenon ", return_tensors="pt").cuda()
x = model.generate(x, max_new_tokens=128).cpu()
print(tok.batch_decode(x))
```

# Conversion

Use `convert_mistral_moe_weights_to_hf.py --input_dir ./input_dir --model_size 7B --output_dir ./output` to convert the original consolidated weights to this HF setup.

Come chat about this in our [Disco(rd)](https://discord.gg/S8W8B5nz3v)! :)