---
language:
- en
library_name: transformers
tags:
- pytorch
- phi
- phi-1_5
- english
datasets:
- ehartford/wizard_vicuna_70k_unfiltered
pipeline_tag: text-generation
inference: true
---
# Phi 1.5 Wizard Vicuna Experimental

Experimental Finetune on Microsoft's [Phi 1.5](https://huggingface.co/microsoft/phi-1_5).
This is highly experimental, only trained on a subset of the 70k Wizard Vicuna dataset and not meant for production use. 

This model also runs reasonably fast on CPU!

Will update with later checkpoints later.

# Prompt Format

ShareGPT / Vicuna (without newlines):
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: This is a question? ASSISTANT: Here is my answer"
```

# Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("jphme/phi-1_5_wizzard_vicuna_uncensored", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("jphme/phi-1_5_wizzard_vicuna_uncensored", trust_remote_code=True, torch_dtype=torch.bfloat16)
prompt_template=("A chat between a curious user and an artificial intelligence assistant. "
"The assistant gives helpful, detailed, and polite answers to the user's questions. "
"USER: {prompt} ASSISTANT:")

inputs = tokenizer(
    prompt_template.format(prompt="What is 1+1?"), 
    return_tensors="pt", return_attention_mask=False).to('cuda')
outputs = model.generate(
    **inputs, max_length=200, 
    do_sample=True, 
    temperature=0.5, 
    top_p=0.95, 
    use_cache=True, 
    repetition_penalty=1.1, 
    #can try to uncomment this if model doesn't end
    #eos_token_id=tokenizer.eos_token_id
)
text = tokenizer.batch_decode(outputs,skip_special_tokens=True)[0]
print(text)
```

The model doesn't stop by itself (this could be an issue due to eos/bos tokens being identical), so you can try to explicitly set the `eos_token_id` (see above) or use a custom stopping criteria.

You can remove `.to("cuda")` to let the model run without a GPU!

# Examples

> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Who was president of the USA in 1995? ASSISTANT: The president of the United States at that time was Bill Clinton (...)


> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Write a python function that returns the fibonacci sequence ASSISTANT:The Fibonacci sequence is a series of numbers where each number after the first two are obtained by adding the two preceding ones together (starting from 0). So the sequence goes 0, 1, 1, 2, 3, 5, 8, 13, 21...
Here's a Python function that generates the nth term in the Fibonacci sequence:
```python
def fibonacci(n):
 if n <= 1:
 return n
 else:
 f = [0,1]
 for i in range(2, n+1):
 f.append(f[-1]+f[-2])
 return f[-1]
```
> This function takes as input `n`, which represents the (...)
(this function works with corrected intendation)

> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What is (500+12)/4? ASSISTANT:The answer to this expression is 7.5. (...)
Some potential here ;-)...

# Thanks

Thanks to Microsoft for releaseing Phi 1.5, [Eric Hartford](https://huggingface.co/ehartford) for releasing the Wizard Vicuna dataset, [Winglian](https://huggingface.co/winglian) for his great work on Axolotl that I use for finetuning and [Teknium](https://huggingface.co/teknium) for some Phi finetuning discussion.


# License

The original licenses of the dataset and model applies. No warranty whatsoever, this model is only intended for research purposes.