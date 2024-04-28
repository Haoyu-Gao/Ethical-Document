---
language:
- en
license: creativeml-openrail-m
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---

Diffuser model for this SD checkpoint:
https://civitai.com/models/25694/epicrealism

**emilianJR/epiCRealism** is the HuggingFace diffuser that you can use with **diffusers.StableDiffusionPipeline()**.

Examples | Examples | Examples
---- | ---- | ----
![](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/d0ecbdfc-b995-4582-91f6-95b214e9d35e/width=1024/02196-1169503035-Best%20quality,%20masterpiece,%20ultra%20high%20res,%20(photorealistic_1.4),%20raw%20photo,%20((monochrome)),%20((grayscale)),%20black%20and%20white%20photo.jpeg) | ![](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/260b9915-c9ca-4461-9d9f-1bec8a5198a4/width=1024/02198-476988828-professional%20portrait%20photograph%20of%20a%20gorgeous%20Norwegian%20girl%20in%20winter%20clothing%20with%20long%20wavy%20blonde%20hair,%20sultry%20flirty%20look,.jpeg) | ![]()
![](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/c8ea1b64-241a-41b8-bc73-6346ffa83eea/width=1024/02197-1830217805-(detailed%20face,%20detailed%20eyes,%20clear%20skin,%20clear%20eyes),%20lotr,%20fantasy,%20elf,%20female,%20full%20body,%20looking%20at%20viewer,%20portrait,%20phot.jpeg) | ![](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/628fe66f-e43b-4ab9-96f0-30b4fa03975a/width=1024/02200-3203910620-RAW%20photo,%20a%2022-year-old-girl,%20upper%20body,%20selfie%20in%20a%20car,%20blue%20hoodie,%20inside%20a%20car,%20driving,%20(lipstick_0.7),%20soft%20lighting,%20h.jpeg) | ![]()
-------


## 🧨 Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).


```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "emilianJR/epiCRealism"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "YOUR PROMPT"
image = pipe(prompt).images[0]

image.save("image.png")
```

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)