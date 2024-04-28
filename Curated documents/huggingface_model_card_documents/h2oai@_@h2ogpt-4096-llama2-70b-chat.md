---
language:
- en
license: llama2
tags:
- facebook
- meta
- pytorch
- llama
- llama-2
- h2ogpt
inference: false
model_type: llama
pipeline_tag: text-generation
---

h2oGPT clone of [Meta's Llama 2 70B Chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf).

Try it live on our [h2oGPT demo](https://gpt.h2o.ai) with side-by-side LLM comparisons and private document chat!

See how it compares to other models on our [LLM Leaderboard](https://evalgpt.ai/)!

See more at [H2O.ai](https://h2o.ai/)


## Model Architecture

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 8192, padding_idx=0)
    (layers): ModuleList(
      (0-79): 80 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=8192, out_features=8192, bias=False)
          (k_proj): Linear4bit(in_features=8192, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=8192, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=8192, out_features=8192, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=8192, out_features=28672, bias=False)
          (up_proj): Linear4bit(in_features=8192, out_features=28672, bias=False)
          (down_proj): Linear4bit(in_features=28672, out_features=8192, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=8192, out_features=32000, bias=False)
)
```