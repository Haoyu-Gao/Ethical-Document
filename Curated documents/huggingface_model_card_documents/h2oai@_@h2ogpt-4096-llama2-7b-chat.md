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

h2oGPT clone of [Meta's Llama 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

Try it live on our [h2oGPT demo](https://gpt.h2o.ai) with side-by-side LLM comparisons and private document chat!

See how it compares to other models on our [LLM Leaderboard](https://evalgpt.ai/)!

See more at [H2O.ai](https://h2o.ai/)


## Model Architecture
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```