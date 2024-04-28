---
language:
- en
license: creativeml-openrail-m
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- art
- artistic
- diffusers
- anime
- dreamshaper
- lcm
duplicated_from: Lykon/AnyLoRA
pipeline_tag: text-to-image
---

# AnyLora

`lykon/AnyLoRA` is a Stable Diffusion model that has been fine-tuned on [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

Please consider supporting me: 
- on [Patreon](https://www.patreon.com/Lykon275)
- or [buy me a coffee](https://snipfeed.co/lykon)

## Diffusers

For more general information on how to run text-to-image models with 🧨 Diffusers, see [the docs](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation).

1. Installation

```
pip install diffusers transformers accelerate
```

2. Run
```py
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch

pipe = AutoPipelineForText2Image.from_pretrained('lykon/AnyLoRA', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=20, generator=generator).images[0]  
image.save("./image.png")
```
