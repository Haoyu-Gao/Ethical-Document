---
language:
- en
license: creativeml-openrail-m
library_name: diffusers
tags:
- stable-diffusion
- stable-diffusion-diffusers
datasets:
- cag/anything-v3-1-dataset
thumbnail: https://huggingface.co/cag/anything-v3-1/resolve/main/example-images/thumbnail.png
pipeline_tag: text-to-image
inference: true
widget:
- text: masterpiece, best quality, 1girl, brown hair, green eyes, colorful, autumn,
    cumulonimbus clouds, lighting, blue sky, falling leaves, garden
  example_title: example 1girl
- text: masterpiece, best quality, 1boy, medium hair, blonde hair, blue eyes, bishounen,
    colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden
  example_title: example 1boy
---

# Anything V3.1

![Anime Girl](https://huggingface.co/cag/anything-v3-1/resolve/main/example-images/thumbnail.png)

Anything V3.1 is a third-party continuation of a latent diffusion model, Anything V3.0. This model is claimed to be a better version of Anything V3.0 with a fixed VAE model and a fixed CLIP position id key. The CLIP reference was taken from Stable Diffusion V1.5. The VAE was swapped using Kohya's merge-vae script and the CLIP was fixed using Arena's stable-diffusion-model-toolkit webui extensions.

Anything V3.2 is supposed to be a resume training of Anything V3.1. The current model has been fine-tuned with a learning rate of 2.0e-6, 50 epochs, and 4 batch sizes on datasets collected from many sources, with 1/4 of them being synthetic datasets. The dataset has been preprocessed using the Aspect Ratio Bucketing Tool so that it can be converted to latents and trained at non-square resolutions. This model is supposed to be a test model to see how the clip fix affects training. Like other anime-style Stable Diffusion models, it also supports Danbooru tags to generate images.

e.g. **_1girl, white hair, golden eyes, beautiful eyes, detail, flower meadow, cumulonimbus clouds, lighting, detailed sky, garden_** 

- Use it with the [`Automatic1111's Stable Diffusion Webui`](https://github.com/AUTOMATIC1111/stable-diffusion-webui) see: ['how-to-use'](#how-to-use)
- Use it with 🧨 [`diffusers`](##🧨Diffusers)


# Model Details

- **Currently maintained by:** Cagliostro Research Lab
- **Model type:** Diffusion-based text-to-image generation model
- **Model Description:** This is a model that can be used to generate and modify anime-themed images based on text prompts. 
- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)
- **Finetuned from model:** Anything V3.1

## How-to-Use
- Download `Anything V3.1` [here](https://huggingface.co/cag/anything-v3-1/resolve/main/anything-v3-1.safetensors), or `Anything V3.2` [here](https://huggingface.co/cag/anything-v3-1/resolve/main/anything-v3-2.safetensors), all model are in `.safetensors` format.
- You need to adjust your prompt using aesthetic tags to get better result, you can use any generic negative prompt or use the following suggested negative prompt to guide the model towards high aesthetic generationse:
```
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
```
- And, the following should also be prepended to prompts to get high aesthetic results:
```
masterpiece, best quality, illustration, beautiful detailed, finely detailed, dramatic light, intricate details
```
## 🧨Diffusers

This model can be used just like any other Stable Diffusion model. For more information, please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion). You can also export the model to [ONNX](https://huggingface.co/docs/diffusers/optimization/onnx), [MPS](https://huggingface.co/docs/diffusers/optimization/mps) and/or [FLAX/JAX](). Pretrained model currently based on Anything V3.1.

You should install dependencies below in order to running the pipeline

```bash
pip install diffusers transformers accelerate scipy safetensors
```
Running the pipeline (if you don't swap the scheduler it will run with the default DDIM, in this example we are swapping it to DPMSolverMultistepScheduler):

```python
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "cag/anything-v3-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "masterpiece, best quality, high quality, 1girl, solo, sitting, confident expression, long blonde hair, blue eyes, formal dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

with autocast("cuda"):
    image = pipe(prompt, 
                 negative_prompt=negative_prompt, 
                 width=512,
                 height=728,
                 guidance_scale=12,
                 num_inference_steps=50).images[0]
    
image.save("anime_girl.png")
```
## Limitation 
This model is overfitted and cannot follow prompts well, even after the text encoder has been fixed. This leads to laziness in prompting, as you will only get good results by typing 1girl. Additionally, this model is anime-based and biased towards anime female characters. It is difficult to generate masculine male characters without providing specific prompts. Furthermore, not much has changed compared to the Anything V3.0 base model, as it only involved swapping the VAE and CLIP models and then fine-tuning for 50 epochs with small scale datasets.

## Example

Here is some cherrypicked samples and comparison between available models

![Anime Girl](https://huggingface.co/cag/anything-v3-1/resolve/main/example-images/1girl.png)
![Anime Boy](https://huggingface.co/cag/anything-v3-1/resolve/main/example-images/1boy.png)
![Aesthetic](https://huggingface.co/cag/anything-v3-1/resolve/main/example-images/aesthetic.png)

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)


## Credit
Public domain.