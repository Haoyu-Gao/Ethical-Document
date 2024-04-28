---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- gpt
- llm
- large language model
- h2o-llmstudio
datasets:
- OpenAssistant/oasst1
inference: false
thumbnail: https://h2o.ai/etc.clientlibs/h2o/clientlibs/clientlib-site/resources/images/favicon.ico
---
# Model Card
## Summary

This model was trained using [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio).
- Base model: [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- Dataset preparation: [OpenAssistant/oasst1](https://github.com/h2oai/h2o-llmstudio/blob/1935d84d9caafed3ee686ad2733eb02d2abfce57/app_utils/utils.py#LL1896C5-L1896C28) personalized


## Usage

To use the model with the `transformers` library on a machine with GPUs, first make sure you have the `transformers`, `accelerate`, `torch` and `einops` libraries installed.

```bash
pip install transformers==4.29.2
pip install accelerate==0.19.0
pip install torch==2.0.0
pip install einops==0.6.1
```

```python
import torch
from transformers import AutoTokenizer, pipeline


tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)

generate_text = pipeline(
    model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_fast=False,
    device_map={"": "cuda:0"},
)

res = generate_text(
    "Why is drinking water so healthy?",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
print(res[0]["generated_text"])
```

You can print a sample prompt after the preprocessing step to see how it is feed to the tokenizer:

```python
print(generate_text.preprocess("Why is drinking water so healthy?")["prompt_text"])
```

```bash
<|prompt|>Why is drinking water so healthy?<|endoftext|><|answer|>
```

Alternatively, you can download [h2oai_pipeline.py](h2oai_pipeline.py), store it alongside your notebook, and construct the pipeline yourself from the loaded model and tokenizer:


```python
import torch
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

res = generate_text(
    "Why is drinking water so healthy?",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
print(res[0]["generated_text"])
```


You may also construct the pipeline from the loaded model and tokenizer yourself and consider the preprocessing steps:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3"  # either local folder or huggingface model name
# Important: The prompt needs to be in the same format the model was trained with.
# You can find an example prompt in the experiment logs.
prompt = "<|prompt|>How are you?<|endoftext|><|answer|>"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
model.cuda().eval()
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

# generate configuration can be modified to your needs
tokens = model.generate(
    **inputs,
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)[0]

tokens = tokens[inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(answer)
```

## Model Architecture

```
RWForCausalLM(
  (transformer): RWModel(
    (word_embeddings): Embedding(65024, 4544)
    (h): ModuleList(
      (0-31): 32 x DecoderLayer(
        (input_layernorm): LayerNorm((4544,), eps=1e-05, elementwise_affine=True)
        (self_attention): Attention(
          (maybe_rotary): RotaryEmbedding()
          (query_key_value): Linear(in_features=4544, out_features=4672, bias=False)
          (dense): Linear(in_features=4544, out_features=4544, bias=False)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): MLP(
          (dense_h_to_4h): Linear(in_features=4544, out_features=18176, bias=False)
          (act): GELU(approximate='none')
          (dense_4h_to_h): Linear(in_features=18176, out_features=4544, bias=False)
        )
      )
    )
    (ln_f): LayerNorm((4544,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4544, out_features=65024, bias=False)
)
```

## Model Configuration

This model was trained using H2O LLM Studio and with the configuration in [cfg.yaml](cfg.yaml). Visit [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) to learn how to train your own large language models.


## Disclaimer

Please read this disclaimer carefully before using the large language model provided in this repository. Your use of the model signifies your agreement to the following terms and conditions.

- Biases and Offensiveness: The large language model is trained on a diverse range of internet text data, which may contain biased, racist, offensive, or otherwise inappropriate content. By using this model, you acknowledge and accept that the generated content may sometimes exhibit biases or produce content that is offensive or inappropriate. The developers of this repository do not endorse, support, or promote any such content or viewpoints.
- Limitations: The large language model is an AI-based tool and not a human. It may produce incorrect, nonsensical, or irrelevant responses. It is the user's responsibility to critically evaluate the generated content and use it at their discretion.
- Use at Your Own Risk: Users of this large language model must assume full responsibility for any consequences that may arise from their use of the tool. The developers and contributors of this repository shall not be held liable for any damages, losses, or harm resulting from the use or misuse of the provided model.
- Ethical Considerations: Users are encouraged to use the large language model responsibly and ethically. By using this model, you agree not to use it for purposes that promote hate speech, discrimination, harassment, or any form of illegal or harmful activities.
- Reporting Issues: If you encounter any biased, offensive, or otherwise inappropriate content generated by the large language model, please report it to the repository maintainers through the provided channels. Your feedback will help improve the model and mitigate potential issues.
- Changes to this Disclaimer: The developers of this repository reserve the right to modify or update this disclaimer at any time without prior notice. It is the user's responsibility to periodically review the disclaimer to stay informed about any changes.

By using the large language model provided in this repository, you agree to accept and comply with the terms and conditions outlined in this disclaimer. If you do not agree with any part of this disclaimer, you should refrain from using the model and any content generated by it.