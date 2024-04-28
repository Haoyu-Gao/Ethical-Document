---
language:
- en
license: openrail++
library_name: diffusers
tags:
- stable-diffusion
- stable-diffusion-diffusers
- stable-diffusion-xl
datasets:
- Linaqruf/animagine-datasets
pipeline_tag: text-to-image
inference:
  parameter:
    negative_prompt: lowres, bad anatomy, bad hands, text, error, missing fingers,
      extra digit, fewer digits, cropped, worst quality, low quality, normal quality,
      jpeg artifacts, signature, watermark, username, blurry
widget:
- text: face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking
    at viewer, upper body, beanie, outdoors, night, turtleneck
  example_title: example 1girl
- text: face focus, bishounen, masterpiece, best quality, 1boy, green hair, sweater,
    looking at viewer, upper body, beanie, outdoors, night, turtleneck
  example_title: example 1boy
---

<style>
  .title-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh; /* Adjust this value to position the title vertically */
  }
  .title {
    font-size: 3em;
    text-align: center;
    color: #333;
    font-family: 'Helvetica Neue', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.5em 0;
    background: transparent;
  }
  .title span {
    background: -webkit-linear-gradient(45deg, #7ed56f, #28b485);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .custom-table {
    table-layout: fixed;
    width: 100%;
    border-collapse: collapse;
    margin-top: 2em;
  }
  .custom-table td {
    width: 50%;
    vertical-align: top;
    padding: 10px;
    box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.15);
  }
  .custom-image {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 10px;
    transition: transform .2s; 
    margin-bottom: 1em;
  }
  .custom-image:hover {
    transform: scale(1.05);
  }
</style>

<h1 class="title"><span>Animagine XL</span></h1>

<table class="custom-table">
  <tr>
    <td>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image1.png">
        <img class="custom-image" src="https://huggingface.co/Linaqruf/animagine-xl/resolve/main/sample_images/image1.png" alt="sample1">
      </a>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image4.png">
        <img class="custom-image" src="https://huggingface.co/Linaqruf/animagine-xl/resolve/main/sample_images/image4.png" alt="sample3">
      </a>
    </td>
    <td>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image2.png">
        <img class="custom-image" src="https://huggingface.co/Linaqruf/animagine-xl/resolve/main/sample_images/image2.png" alt="sample2">
      </a>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image3.png">
        <img class="custom-image" src="https://huggingface.co/Linaqruf/animagine-xl/resolve/main/sample_images/image3.png" alt="sample4">
      </a>
    </td>
  </tr>
</table>

<hr>

## Overview

**Animagine XL** is a high-resolution, latent text-to-image diffusion model. The model has been fine-tuned using a learning rate of `4e-7` over 27000 global steps with a batch size of 16 on a curated dataset of superior-quality anime-style images. This model is derived from Stable Diffusion XL 1.0.

- Use it with the [`Stable Diffusion Webui`](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Use it with 🧨 [`diffusers`](https://huggingface.co/docs/diffusers/index)
- Use it with the [`ComfyUI`](https://github.com/comfyanonymous/ComfyUI) **(recommended)**

Like other anime-style Stable Diffusion models, it also supports Danbooru tags to generate images.

e.g. _**face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck**_

<hr>

## Features

1. High-Resolution Images: The model trained with 1024x1024 resolution. The model is trained using [NovelAI Aspect Ratio Bucketing Tool](https://github.com/NovelAI/novelai-aspect-ratio-bucketing) so that it can be trained at non-square resolutions.
2. Anime-styled Generation: Based on given text prompts, the model can create high quality anime-styled images.
3. Fine-Tuned Diffusion Process: The model utilizes a fine-tuned diffusion process to ensure high quality and unique image output.

<hr>

## Model Details

- **Developed by:** [Linaqruf](https://github.com/Linaqruf)
- **Model type:** Diffusion-based text-to-image generative model
- **Model Description:** This is a model that can be used to generate and modify high quality anime-themed images based on text prompts. 
- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)
- **Finetuned from model:** [Stable Diffusion XL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

<hr>

## How to Use:
- Download `Animagine XL` [here](https://huggingface.co/Linaqruf/animagine-xl/resolve/main/animagine-xl.safetensors), the model is in `.safetensors` format.
- You need to use Danbooru-style tag as prompt instead of natural language, otherwise you will get realistic result instead of anime
- You can use any generic negative prompt or use the following suggested negative prompt to guide the model towards high aesthetic generationse:
```
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
```
- And, the following should also be prepended to prompts to get high aesthetic results:
```
masterpiece, best quality
```
- Use this cheat sheet to find the best resolution:
```
768 x 1344: Vertical (9:16)
915 x 1144: Portrait (4:5)
1024 x 1024: Square (1:1)
1182 x 886: Photo (4:3)
1254 x 836: Landscape (3:2)
1365 x 768: Widescreen (16:9)
1564 x 670: Cinematic (21:9)
```
<hr>

## Gradio & Colab

We also support a [Gradio](https://github.com/gradio-app/gradio) Web UI and Colab with Diffusers to run **Animagine XL**:
[![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/Linaqruf/Animagine-XL)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/Linaqruf/animagine-xl/blob/main/Animagine_XL_demo.ipynb)


## 🧨 Diffusers 

Make sure to upgrade diffusers to >= 0.18.2:
```
pip install diffusers --upgrade
```

In addition make sure to install `transformers`, `safetensors`, `accelerate` as well as the invisible watermark:
```
pip install invisible_watermark transformers accelerate safetensors
```

Running the pipeline (if you don't swap the scheduler it will run with the default **EulerDiscreteScheduler** in this example we are swapping it to **EulerAncestralDiscreteScheduler**:
```py
import torch
from torch import autocast
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

model = "Linaqruf/animagine-xl"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
    )

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    width=1024,
    height=1024,
    guidance_scale=12,
    target_size=(1024,1024),
    original_size=(4096,4096),
    num_inference_steps=50
    ).images[0]

image.save("anime_girl.png")
```
<hr>

## Limitation 
This model inherit Stable Diffusion XL 1.0 [limitation](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0#limitations)
