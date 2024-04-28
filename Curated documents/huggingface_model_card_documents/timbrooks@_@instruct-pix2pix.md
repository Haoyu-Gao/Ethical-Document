---
license: mit
tags:
- image-to-image
---

# InstructPix2Pix: Learning to Follow Image Editing Instructions
GitHub: https://github.com/timothybrooks/instruct-pix2pix
<img src='https://instruct-pix2pix.timothybrooks.com/teaser.jpg'/>



## Example

To use `InstructPix2Pix`, install `diffusers` using `main` for now. The pipeline will be available in the next release

```bash
pip install diffusers accelerate safetensors transformers
```

```python
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = "turn him into cyborg"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]
```