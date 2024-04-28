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
duplicated_from: lykon/dreamshaper-8
pipeline_tag: text-to-image
---

# Dreamshaper 8

`lykon/dreamshaper-8` is a Stable Diffusion model that has been fine-tuned on [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

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

pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-8', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"

generator = torch.manual_seed(33)
image = pipe(prompt, generator=generator, num_inference_steps=25).images[0]  
image.save("./image.png")
```

![](./image.png)

## Notes

- **Version 8** focuses on improving what V7 started. Might be harder to do photorealism compared to realism focused models, as it might be hard to do anime compared to anime focused models, but it can do both pretty well if you're skilled enough. Check the examples!
- **Version 7** improves lora support, NSFW and realism. If you're interested in "absolute" realism, try AbsoluteReality.
- **Version 6** adds more lora support and more style in general. It should also be better at generating directly at 1024 height (but be careful with it). 6.x are all improvements.
- **Version 5** is the best at photorealism and has noise offset.
- **Version 4** is much better with anime (can do them with no LoRA) and booru tags. It might be harder to control if you're used to caption style, so you might still want to use version 3.31. V4 is also better with eyes at lower resolutions. Overall is like a "fix" of V3 and shouldn't be too much different.