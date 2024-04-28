---
language:
- ja
- en
license: llama2
datasets:
- databricks/databricks-dolly-15k
- kunishou/databricks-dolly-15k-ja
- izumi-lab/llm-japanese-dataset
thumbnail: https://github.com/rinnakk/japanese-pretrained-models/blob/master/rinna.png
inference: false
---

# `rinna/youri-7b-chat-gptq`

![rinna-icon](./rinna.png)

# Overview
`rinna/youri-7b-chat-gptq` is the quantized model for [`rinna/youri-7b-chat`](https://huggingface.co/rinna/youri-7b-chat) using AutoGPTQ. The quantized version is 4x smaller than the original model and thus requires less memory and provides faster inference.

* **Model architecture**

    Refer to the [original model](https://huggingface.co/rinna/youri-7b-chat) for architecture details.

* **Fine-tuning**
    
    Refer to the [original model](https://huggingface.co/rinna/youri-7b-chat) for fine-tuning details.

* **Authors**
    
    - [Toshiaki Wakatsuki](https://huggingface.co/t-w)
    - [Tianyu Zhao](https://huggingface.co/tianyuz)
    - [Kei Sawada](https://huggingface.co/keisawada)

---

# Benchmarking

Please refer to [rinna's LM benchmark page](https://rinnakk.github.io/research/benchmarks/lm/index.html).

---

# How to use the model

~~~~python
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True)

instruction = "次の日本語を英語に翻訳してください。"
input = "自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。"

context = [
    {
        "speaker": "設定",
        "text": instruction
    },
    {
        "speaker": "ユーザー",
        "text": input
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in context
]
prompt = "\n".join(prompt)
prompt = (
    prompt
    + "\n"
    + "システム: "
)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

output = output[len(prompt):-len("</s>")].strip()
input = "大規模言語モデル（だいきぼげんごモデル、英: large language model、LLM）は、多数のパラメータ（数千万から数十億）を持つ人工ニューラルネットワークで構成されるコンピュータ言語モデルで、膨大なラベルなしテキストを使用して自己教師あり学習または半教師あり学習によって訓練が行われる。"

context.extend([
    {
        "speaker": "システム",
        "text": output
    },
    {
        "speaker": "ユーザー",
        "text": input
    }
])
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in context
]
prompt = "\n".join(prompt)
prompt = (
    prompt
    + "\n"
    + "システム: "
)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)
~~~~

---

# Tokenization
The model uses the original llama-2 tokenizer.

---

# How to cite
~~~
@misc{RinnaYouri7bChatGPTQ, 
    url={https://huggingface.co/rinna/youri-7b-chat-gptq}, 
    title={rinna/youri-7b-chat-gptq}, 
    author={Wakatsuki, Toshiaki and Zhao, Tianyu and Sawada, Kei}
}
~~~
---

# License
[The llama2 license](https://ai.meta.com/llama/license/)