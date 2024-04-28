---
license: openrail++
library_name: diffusers
tags:
- lora
- text-to-image
base_model: segmind/SSD-1B
inference: false
---

# Latent Consistency Model (LCM) LoRA: SSD-1B

Latent Consistency Model (LCM) LoRA was proposed in [LCM-LoRA: A universal Stable-Diffusion Acceleration Module](https://arxiv.org/abs/2311.05556) 
by *Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu et al.*

It is a distilled consistency adapter for [`segmind/SSD-1B`](https://huggingface.co/segmind/SSD-1B) that allows
to reduce the number of inference steps to only between **2 - 8 steps**.

| Model                                                                      | Params / M | 
|----------------------------------------------------------------------------|------------|
| [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)   | 67.5       |
| [**lcm-lora-ssd-1b**](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b)   | **105**        |
| [lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl) | 197M  |

## Usage

LCM-LoRA is supported in 🤗 Hugging Face Diffusers library from version v0.23.0 onwards. To run the model, first 
install the latest version of the Diffusers library as well as `peft`, `accelerate` and `transformers`.
audio dataset from the Hugging Face Hub:

```bash
pip install --upgrade pip
pip install --upgrade diffusers transformers accelerate peft
```

### Text-to-Image

Let's load the base model `segmind/SSD-1B` first. Next, the scheduler needs to be changed to [`LCMScheduler`](https://huggingface.co/docs/diffusers/v0.22.3/en/api/schedulers/lcm#diffusers.LCMScheduler) and we can reduce the number of inference steps to just 2 to 8 steps.
Please make sure to either disable `guidance_scale` or use values between 1.0 and 2.0.

```python
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

model_id = "segmind/SSD-1B"
adapter_id = "latent-consistency/lcm-lora-ssd-1b"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()


prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# disable guidance_scale by passing 0
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]
```

![](./image.png)

### Image-to-Image

Works as well! TODO docs

### Inpainting

Works as well! TODO docs

### ControlNet

Works as well! TODO docs

### T2I Adapter

Works as well! TODO docs

## Speed Benchmark

TODO

## Training

TODO