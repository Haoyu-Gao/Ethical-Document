---
library_name: transformers
pipeline_tag: text-generation
inference: true
widget:
- text: Hello!
  example_title: Hello world
  group: Python
---

# yujiepan/llama-2-tiny-3layers-random

This model is **randomly initialized**, using the config from [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/yujiepan/llama-2-tiny-3layers-random/blob/main/config.json) but with the following modifications:

```json
{
  "hidden_size": 8,
  "intermediate_size": 32,
  "num_attention_heads": 2,
  "num_hidden_layers": 3,
  "num_key_value_heads": 2,
}

```