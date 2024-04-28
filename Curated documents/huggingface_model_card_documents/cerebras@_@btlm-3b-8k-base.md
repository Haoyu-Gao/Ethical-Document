---
language:
- en
license: apache-2.0
tags:
- pytorch
- causal-lm
- Cerebras
- BTLM
datasets:
- cerebras/SlimPajama-627B
inference: false
pipeline_tag: text-generation
---

# BTLM-3B-8k-base

[Bittensor Language Model (BTLM-3B-8k-base)](https://www.cerebras.net/blog/btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model/) is a 3 billion parameter language model with an 8k context length trained on 627B tokens of [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B). BTLM-3B-8k-base sets a new standard for 3B parameter models, outperforming models trained on hundreds of billions more tokens and achieving comparable performance to open 7B parameter models. BTLM-3B-8k-base can also be quantized to 4-bit to fit in devices with as little as 3GB of memory. The model is made available with an Apache 2.0 license for commercial use.

BTLM was trained by [Cerebras](https://www.cerebras.net/) in partnership with [Opentensor](https://opentensor.ai/) on the newly unveiled [Condor Galaxy 1 (CG-1) supercomputer](https://www.cerebras.net/blog/introducing-condor-galaxy-1-a-4-exaflop-supercomputer-for-generative-ai/), the first public deliverable of the G42-Cerebras strategic partnership. 

BTLM-3B-8k was trained with a similar architecture to [CerebrasGPT](https://arxiv.org/abs/2304.03208) with the addition of [SwiGLU](https://arxiv.org/abs/2002.05202) nonlinearity, [ALiBi](https://arxiv.org/abs/2108.12409) position embeddings, and [maximal update parameterization (muP)](https://arxiv.org/abs/2203.03466). The model was trained for 1 epoch of SlimPajama-627B. 75% of training was performed with 2k sequence length. The final 25% of training was performed at 8k sequence length to enable long sequence applications

Read [our paper](https://arxiv.org/abs/2309.11568) for more details!

## BTLM-3B-8k Highlights

BTLM-3B-8k-base:
- **Licensed for commercial use** (Apache 2.0).
- **[State of the art 3B parameter model](#performance-vs-3b-models)**.
- **Provides 7B model performance in a 3B model** via performance enhancements from [ALiBi](https://arxiv.org/abs/2108.12409), [SwiGLU](https://arxiv.org/abs/2002.05202), [maximal update parameterization (muP)](https://arxiv.org/abs/2203.03466) and the the extensively deduplicated and cleaned [SlimPajama-627B dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B).
- **[Fits in devices with as little as 3GB of memory](#memory-requirements) when quantized to 4-bit**.
- **One of few 3B models that supports 8k sequence length** thanks to ALiBi. 
- **Requires 71% fewer training FLOPs, has 58% smaller memory footprint** for inference than comparable 7B models.

## Usage
*Note: Transformers does not support muP for all models, so BTLM-3B-8k-base requires a custom model class. This causes a situation where users must either (1) enable `trust_remote_code=True` when loading the model or (2) acknowledge the warning about code execution upon loading the model.*

#### With generate():
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cerebras/btlm-3b-8k-base")
model = AutoModelForCausalLM.from_pretrained("cerebras/btlm-3b-8k-base", trust_remote_code=True, torch_dtype="auto")

# Set the prompt for generating text
prompt = "Albert Einstein was known for "

# Tokenize the prompt and convert to PyTorch tensors
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text using the model
outputs = model.generate(
    **inputs,
    num_beams=5,
    max_new_tokens=50,
    early_stopping=True,
    no_repeat_ngram_size=2
)

# Convert the generated token IDs back to text
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print the generated text
print(generated_text[0])
```

#### With pipeline:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cerebras/btlm-3b-8k-base")
model = AutoModelForCausalLM.from_pretrained("cerebras/btlm-3b-8k-base", trust_remote_code=True, torch_dtype="auto")

# Set the prompt for text generation
prompt = """Isaac Newton was a """

# Create a text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text using the pipeline
generated_text = pipe(
    prompt, 
    max_length=50, 
    do_sample=False, 
    no_repeat_ngram_size=2)[0]

# Print the generated text
print(generated_text['generated_text'])
```

## Evaluations and Comparisons to Other Models

### Memory Requirements
![figure_1_image](./figure_1_memory_footprint.png)
Figure 1. Memory requirements of different model sizes and quantization schemes

### Quality, Training Cost, Memory Footprint, Inference Speed
![figure_2_image](./figure_2_half_the_size_twice_the_speed.png)
Figure 2: Comparisons of quality, memory footprint & inference cost between BTLM-3B-8K and 7B model families.  

### Performance vs 3B models
![table_1_image](./table_1_downstream_performance_3b.png)
Table 1: Performance at 3B model size. Detailed down-stream tasks comparisons. MMLU task performance is reported using 5-shot, other tasks are 0-shot. 

![figure_3_image](./figure_3_performance_vs_3b_models.png)
Figure 3: Performance at 3B model size

### Performance vs 7B models
![table_2_image](./table_2_downstream_performance_7b.png)
Table 2: Performance at 7B model size. Detailed down-stream tasks comparisons. MMLU task performance is reported using 5-shot, everything else is 0-shot. 

![figure_4_image](./figure_4_performance_vs_7b_models.jpg)
Figure 4: Performance at 7B model size

## Long Sequence Lengths
To enable long sequence applications, we use ALiBi position embeddings and trained on 470B tokens at the context length of 2,048 followed by 157B of tokens trained at 8,192 context length. To assess BTLM’s long sequence capability, we evaluate it on SlimPajama test set with 32,768 context length and plot loss at each token position. Although ALiBi allows extrapolation in theory, 2,048 context length training alone does not extrapolate well in practice. Thankfully variable sequence length training allows for substantially improved extrapolation. BTLM-3B extrapolates well up to 10k context length but the performance degrades slightly beyond this.

![figure_5_image](./figure_5_xentropy_with_sequence_lengths.svg)
Figure 5: BTLM-3B model's cross-entropy evaluation on the SlimPajama’s test set. Inference performed on the extrapolated sequence length of 32,768 tokens.

## Model Details
- Developed by: [Cerebras Systems](https://www.cerebras.net/) and [Opentensor](https://opentensor.ai/) with generous support from [G42 Cloud](https://www.g42cloud.com/) and [IIAI](https://www.inceptioniai.org/en/)
- License: Apache 2.0
- Model type: Decoder-only Language Model
- Architecture: GPT-2 style architecture with SwiGLU, ALiBi, and muP
- Data set: SlimPajama-627B
- Tokenizer: Byte Pair Encoding
- Vocabulary Size: 50257
- Sequence Length: 8192
- Optimizer: AdamW
- Positional Encoding: ALiBi
- Language: English
- Learn more: [BTLM-3B-8k blog](https://www.cerebras.net/blog/btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model/)
- Paper: [BTLM-3B-8K: 7B Parameter Performance in a 3B Parameter Model](https://arxiv.org/abs/2309.11568)

## To continue training with PyTorch and Maximal Update Parameterization

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("cerebras/btlm-3b-8k-base", trust_remote_code=True)

# Get the parameter groups for the muP optimizer
param_groups = model.get_mup_param_groups(lr=1e-3, weight_decay=0.1)

# Set up the optimizer using AdamW with muP parameters
optimizer = torch.optim.AdamW(
    param_groups,
    betas=(0.9, 0.95),
    eps=1e-8
)
```

Ensure the following muP parameters are passed in your config, otherwise your model will default to standard parameterization
- `mup_width_scale: <float>`
- `mup_embeddings_scale: <float>`
- `mup_output_alpha: <float>`
- `mup_scale_qk_dot_by_d: true`

## To extend the context length with Position Interpolation

### During inference (without fine-tuning):
It's possible to extend the context length to 2x the training context length without degradation in performance using dynamic linear scaling. Dynamic linear scaling adjusts the slopes of ALiBi with a factor of `input_seq_len/train_seq_len` when `input_seq_len` is larger than `train_seq_len`. Check the details in our paper [Position Interpolation Improves ALiBi Extrapolation](https://arxiv.org/abs/2310.13017). To enable dynamic linear scaling, update `config.json` as follows:
```json
  # update `n_positions` with the maximum context length will be 
  # encountered during inference (e.g. 16384 tokens)
  "n_positions": 16384,

  # specify `train_seq_len` in `alibi_scaling` parameter
  "alibi_scaling": {
    "type": "linear",
    "train_seq_len": 8192
  }
```

### Using fine-tuning + position interpolation:
Performing fine-tuning with position interpolation can help achieve greater extrapolation lengths. The scaling factor should be fixed to `finetuning_seq_len/train_seq_len`. To enable fixed linear scaling, update `config.json` as follows:
```json
  # update `n_positions` with the fine-tuning context length (e.g. 32768 tokens)
  "n_positions": 32768,

  # specify the scaling `factor` in `alibi_scaling` parameter
  "alibi_scaling": {
    "type": "linear",
    "factor": 4.0
  }
```

## Uses and Limitations

### Intended Use
The primary intended use is to further research into large language models. BTLM-3B-8k-base can be used as a foundation model for NLP, applications, ethics, and alignment research. We release these models with a fully permissive Apache license for the community to use freely.

You may fine-tune and adapt BTLM-3B-8k-base model via either Cerebras [Model Studio](https://www.cerebras.net/product-cloud/) or third-party libraries. Further safety-related testing and mitigations should be applied before using the BTLM-3B-8k-base in production downstream applications.

### Out of Scope Use
BTLM-3B-8k-base was trained on SlimPajama, with primarily English language, and is not recommended for machine translation tasks. BTLM-3B-8k-base has not been tuned for instruction-following or chat-based use cases.

### Generation Samples
```
Prompt: What is a manometer?"
Generation: "A manometric device is an instrument that measures the pressure of a fluid."
```

```
Prompt: "What Robert Oppenheimer is famous for?"
Generation: “Robert Opperman was a physicist who was the director of the Manhattan Project, which was responsible for the development of nuclear weapons.”
```

```
Prompt: "What was Albert Einstein known for?"
Generation: “Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics).”
```

```
Prompt: “Who is Isaac Newton?”
Generation: “Isaac Newton was an English physicist, mathematician, astronomer, alchemist, and theologian. He is widely recognized as one of the most influential scientists of all time and as a key figure in the scientific revolution.”
```

```
Prompt: "Where does the blood get pumped to after it leaves the right ventricle?"
Generation: “The right atrium receives the deoxygenated blood from the pulmonary artery. The right side of the heart is the only side that receives de-oxygened...”
```

```
Prompt: "Peter lost his wallet after attending a local concert. To help him, Bob gave David 10 dollars. David then passed it on to Peter. Now what is the total amount of money Peter has?"
Generation: “A. $10”
```

## Risk, Bias, Ethical Considerations
- **Human life:** The outputs from this model may or may not align with human values. The risk needs to be thoroughly investigated before deploying this model in a production environment where it can directly impact human life.
- **Risks and harms:** There may be distributional bias in the [RedPajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) that can manifest in various forms in the downstream model deployment. There are other risks associated with large language models such as amplifying stereotypes, memorizing training data, or revealing private or secure information.

## Acknowledgements
We are thankful to all Cerebras engineers that made this work possible.

We would like to acknowledge the generous support of G42 Cloud and the Inception Institute of Artificial Intelligence for providing compute time on Condor Galaxy 1.