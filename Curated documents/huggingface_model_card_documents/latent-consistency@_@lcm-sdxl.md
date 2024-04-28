---
license: openrail++
library_name: diffusers
tags:
- text-to-image
base_model: stabilityai/stable-diffusion-xl-base-1.0
inference: false
---

# Latent Consistency Model (LCM): SDXL

Latent Consistency Model (LCM) was proposed in [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378) 
by *Simian Luo, Yiqin Tan et al.* and [Simian Luo](https://huggingface.co/SimianLuo), [Suraj Patil](https://huggingface.co/valhalla), and [Daniel Gu](https://huggingface.co/dg845)
succesfully applied the same approach to create LCM for SDXL.

This checkpoint is a LCM distilled version of [`stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) that allows
to reduce the number of inference steps to only between **2 - 8 steps**.


## Usage

LCM SDXL is supported in 🤗 Hugging Face Diffusers library from version v0.23.0 onwards. To run the model, first 
install the latest version of the Diffusers library as well as `peft`, `accelerate` and `transformers`.
audio dataset from the Hugging Face Hub:

```bash
pip install --upgrade pip
pip install --upgrade diffusers transformers accelerate peft
```

### Text-to-Image

The model can be loaded with it's base pipeline `stabilityai/stable-diffusion-xl-base-1.0`. Next, the scheduler needs to be changed to [`LCMScheduler`](https://huggingface.co/docs/diffusers/v0.22.3/en/api/schedulers/lcm#diffusers.LCMScheduler) and we can reduce the number of inference steps to just 2 to 8 steps.
Please make sure to either disable `guidance_scale` or use values between 1.0 and 2.0.

```python
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "a close-up picture of an old man standing in the rain"

image = pipe(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
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