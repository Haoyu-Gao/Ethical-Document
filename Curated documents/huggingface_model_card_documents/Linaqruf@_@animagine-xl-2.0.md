---
language:
- en
license: openrail++
library_name: diffusers
tags:
- text-to-image
- stable-diffusion
- safetensors
- stable-diffusion-xl
base_model: stabilityai/stable-diffusion-xl-base-1.0
widget:
- text: face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking
    at viewer, upper body, beanie, outdoors, night, turtleneck
  parameter:
    negative_prompt: lowres, bad anatomy, bad hands, text, error, missing fingers,
      extra digit, fewer digits, cropped, worst quality, low quality, normal quality,
      jpeg artifacts, signature, watermark, username, blurry
  output:
    url: https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/cR_r0k0CSapphAaFrkN1h.png
  example_title: 1girl
- text: face focus, bishounen, masterpiece, best quality, 1boy, green hair, sweater,
    looking at viewer, upper body, beanie, outdoors, night, turtleneck
  parameter:
    negative_prompt: lowres, bad anatomy, bad hands, text, error, missing fingers,
      extra digit, fewer digits, cropped, worst quality, low quality, normal quality,
      jpeg artifacts, signature, watermark, username, blurry
  output:
    url: https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/EteXoZZN4SwlkqfbPpNak.png
  example_title: 1boy
---

<style>
  .title-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh; /* Adjust this value to position the title vertically */
  }
  
  .title {
    font-size: 2.5em;
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
    box-shadow: 0px 0px 0px 0px rgba(0, 0, 0, 0.15);
  }

  .custom-image-container {
    position: relative;
    width: 100%;
    margin-bottom: 0em;
    overflow: hidden;
    border-radius: 10px;
    transition: transform .7s;
    /* Smooth transition for the container */
  }

  .custom-image-container:hover {
    transform: scale(1.05);
    /* Scale the container on hover */
  }

  .custom-image {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 10px;
    transition: transform .7s;
    margin-bottom: 0em;
  }

  .nsfw-filter {
    filter: blur(8px); /* Apply a blur effect */
    transition: filter 0.3s ease; /* Smooth transition for the blur effect */
  }

  .custom-image-container:hover .nsfw-filter {
    filter: none; /* Remove the blur effect on hover */
  }
  
  .overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    color: white;
    width: 100%;
    height: 40%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    font-size: 1vw;
    font-style: bold;
    text-align: center;
    opacity: 0;
    /* Keep the text fully opaque */
    background: linear-gradient(0deg, rgba(0, 0, 0, 0.8) 60%, rgba(0, 0, 0, 0) 100%);
    transition: opacity .5s;
  }
  .custom-image-container:hover .overlay {
    opacity: 1;
    /* Make the overlay always visible */
  }
  .overlay-text {
    background: linear-gradient(45deg, #7ed56f, #28b485);
    -webkit-background-clip: text;
    color: transparent;
    /* Fallback for browsers that do not support this effect */
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    /* Enhanced text shadow for better legibility */
    
  .overlay-subtext {
    font-size: 0.75em;
    margin-top: 0.5em;
    font-style: italic;
  }
    
  .overlay,
  .overlay-subtext {
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  }
    
</style>

<h1 class="title">
  <span>Animagine XL 2.0</span>
</h1>
<table class="custom-table">
  <tr>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/fmkK9WYAPgwbrDcKOybBZ.png" alt="sample1">
      </div>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/TFaH_13XbFh0_NSn4Tzav.png" alt="sample4">
      </div>
    </td>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/twkZ4xvmUBTWZZ88DG0v-.png" alt="sample2">
      </div>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/5LyRRqLwt73u-eOy1HZ_7.png" alt="sample3">
    </td>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/f8aLXc_Slewo7iVxlE246.png" alt="sample1">
      </div>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/PYI5I7VR_zdEZUidn8fIr.png" alt="sample4">
      </div>
    </td>
  </tr>
</table>

## Overview 

**Animagine XL 2.0** is an advanced latent text-to-image diffusion model designed to create high-resolution, detailed anime images. It's fine-tuned from Stable Diffusion XL 1.0 using a high-quality anime-style image dataset. This model, an upgrade from Animagine XL 1.0, excels in capturing the diverse and distinct styles of anime art, offering improved image quality and aesthetics.

## Model Details

- **Developed by:** [Linaqruf](https://github.com/Linaqruf)
- **Model type:** Diffusion-based text-to-image generative model
- **Model Description:** This is a model that excels in creating detailed and high-quality anime images from text descriptions. It's fine-tuned to understand and interpret a wide range of descriptive prompts, turning them into stunning visual art.
- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)
- **Finetuned from model:** [Stable Diffusion XL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

## LoRA Collection

The Animagine XL 2.0 model is complemented by an impressive suite of LoRA (Low-Rank Adaptation) adapters, each designed to imbue the generated images with unique stylistic attributes. This collection of adapters allows users to customize the aesthetic of their creations to match specific art styles, ranging from the vivid and bright Pastel Style to the intricate and ornate Anime Nouveau. 

<table class="custom-table">
  <tr>
    <td>
      <div class="custom-image-container">
        <a href="https://huggingface.co/Linaqruf/style-enhancer-xl-lora">
          <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/7k2c5pW6zMpOiuW9kVsrs.png" alt="sample1">
          <div class="overlay"> Style Enhancer </div>
        </a>
      </div>
    </td>
    <td>
      <div class="custom-image-container">
        <a href="https://huggingface.co/Linaqruf/anime-detailer-xl-lora">
          <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/2yAWKA84ux1wfzaMD3cNu.png" alt="sample1">
          <div class="overlay"> Anime Detailer </div>
        </a>
      </div>
    </td>
    <td>
      <div class="custom-image-container">
        <a href="https://huggingface.co/Linaqruf/sketch-style-xl-lora">
          <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/Iv6h6wC4HTq0ue5UABe_W.png" alt="sample1">
          <div class="overlay"> Sketch Style </div>
        </a>
      </div>
    </td>
    <td>
      <div class="custom-image-container">
        <a href="https://huggingface.co/Linaqruf/pastel-style-xl-lora">
          <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/0Bu6fj33VHC2rTXoD-anR.png" alt="sample1">
          <div class="overlay"> Pastel Style </div>
        </a>
      </div>
    </td>
    <td>
      <div class="custom-image-container">
        <a href="https://huggingface.co/Linaqruf/anime-nouveau-xl-lora">
          <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/Mw_U_1VcrcBGt-i6Lu06d.png" alt="sample1">
          <div class="overlay"> Anime Nouveau </div>
        </a>
      </div>
    </td>
  </tr>
</table>

## Gradio & Colab Integration

Animagine XL is accessible via [Gradio](https://github.com/gradio-app/gradio) Web UI and Google Colab, offering user-friendly interfaces for image generation:

- **Gradio Web UI**: [![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/Linaqruf/Animagine-XL)
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/Linaqruf/animagine-xl/blob/main/Animagine_XL_demo.ipynb)

## 🧨 Diffusers Installation

Ensure the installation of the latest `diffusers` library, along with other essential packages:

```bash
pip install diffusers --upgrade
pip install transformers accelerate safetensors
```

The following Python script demonstrates how to do inference with Animagine XL 2.0. The default scheduler in the model config is EulerAncestralDiscreteScheduler, but it can be explicitly defined for clarity.

```py
import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Linaqruf/animagine-xl-2.0", 
    vae=vae,
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

# Define prompts and generate image
prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    width=1024,
    height=1024,
    guidance_scale=12,
    num_inference_steps=50
).images[0]

```

## Usage Guidelines

### Prompt Guidelines

Animagine XL 2.0 responds effectively to natural language descriptions for image generation. For example:
```
A girl with mesmerizing blue eyes looks at the viewer. Her long, white hair is adorned with blue butterfly hair ornaments.
```

However, to achieve optimal results, it's recommended to use Danbooru-style tagging in your prompts, as the model is trained with images labeled using these tags. For instance:
```
1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck
```

This model incorporates quality and rating modifiers during dataset processing, influencing image generation based on specified criteria:


### Quality Modifiers

| Quality Modifier | Score Criterion |
| ---------------- | --------------- |
| masterpiece      | >150            |
| best quality     | 100-150         |
| high quality     | 75-100          |
| medium quality   | 25-75           |
| normal quality   | 0-25            |
| low quality      | -5-0            |
| worst quality    | <-5             |

### Rating Modifiers

| Rating Modifier | Rating Criterion |
| --------------- | ---------------- |
| -               | general          |
| -               | sensitive        |
| nsfw            | questionable     |
| nsfw            | explicit         |

To guide the model towards generating high-aesthetic images, use negative prompts like:

```
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
```
For higher quality outcomes, prepend prompts with:

```
masterpiece, best quality
```

### Quality Tags Comparison

This table presents a detailed comparison to illustrate how training quality tags can significantly influence the outcomes of generative results. It showcases various attributes, both positive and negative, demonstrating the impact of quality tags in steering the generation of visual content.

<table class="custom-table">
    <tr>
        <th colspan="6" align="center"> Quality Tags Comparison </th>
    </tr>
        <tr>
            <td colspan="1">Prompt</td>
            <td colspan="5" align="center" style="font-style: italic">"1girl, fu xuan, honkai:star rail, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"</td>
        </tr>
    <tr>
        <td>Positive</td>
        <td>-</td>
        <td>masterpiece, best quality</td>
        <td>-</td>
        <td>masterpiece, best quality</td>
        <td>masterpiece, best quality</td>
    </tr>
    <tr>
        <td>Negative</td>
        <td>-</td>
        <td>-</td>
        <td>worst quality, low quality, normal quality</td>
        <td>worst quality, low quality, normal quality</td>
        <td>lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry</td>
    </tr>
    <tr>
        <td></td>
        <td>
            <div class="custom-image-container">
                <a href="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/6Jgm3iii23ZMHVAJcR02u.png" target="_blank">
                    <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/6Jgm3iii23ZMHVAJcR02u.png" alt="Comparison 1">
                </a>
            </div>
        </td>
        <td>
            <div class="custom-image-container">
                <a href="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/vLYdEN3u5GnIaTDiPT-Nw.png" target="_blank">
                    <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/vLYdEN3u5GnIaTDiPT-Nw.png" alt="Comparison 2">
                </a>
            </div>
        </td>
        <td>
            <div class="custom-image-container">
                <a href="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/4jw_6xjEWmcqwPNFp6ktC.png" target="_blank">
                    <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/4jw_6xjEWmcqwPNFp6ktC.png" alt="Comparison 3">
                </a>
            </div>
        </td>
        <td>
            <div class="custom-image-container">
                <a href="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/x7SNaPLKJXm1ZtoKIYiHs.png" target="_blank">
                    <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/x7SNaPLKJXm1ZtoKIYiHs.png" alt="Comparison 4">
                </a>
            </div>
        </td>
        <td>
            <div class="custom-image-container">
                <a href="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/5HnkLvrahnqdL28_GegxI.png" target="_blank">
                    <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/5HnkLvrahnqdL28_GegxI.png" alt="Comparison 5">
                </a>
            </div>
        </td>
    </tr>

</table>

## Examples
<table class="custom-table">
  <tr>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/m6BGzrJgYTb9QrZprVAqZ.png" alt="sample1">
        <div class="overlay" style="font-size: 1vw; font-style: bold;"> Twilight Contemplation <div class="overlay-subtext" style="font-size: 0.75em; font-style: italic;">"Stelle, Amidst Shooting Stars and Mountain Silhouettes"</div>
        </div>
      </div>
    </td>
  </tr>
</table>

<details>
  <summary>Generation Parameter</summary>
  <pre>
{
  "prompt": "cinematic photo (masterpiece), (best quality), (ultra-detailed), stelle, honkai: star rail, official art, 1girl, solo, gouache, starry sky, mountain, long hair, hoodie, shorts, sneakers, yellow eyes, tsurime, sitting on a rock, stargazing, milky way, shooting star, tranquil night., illustration, disheveled hair, detailed eyes, perfect composition, moist skin, intricate details, earrings . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
  "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, uglylongbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality",
  "resolution": "832 x 1216",
  "guidance_scale": 12,
  "num_inference_steps": 50,
  "seed": 1082676886,
  "sampler": "Euler a",
  "enable_lcm": false,
  "sdxl_style": "Photographic",
  "quality_tags": "Heavy",
  "refine_prompt": false,
  "use_lora": null,
  "use_upscaler": {
    "upscale_method": "nearest-exact",
    "upscaler_strength": 0.55,
    "upscale_by": 1.5,
    "new_resolution": "1248 x 1824"
  },
  "datetime": "2023-11-25 06:42:21.342459"
}
  </pre>
</details>

<table class="custom-table">
  <tr>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/7f6BZyn1m30qHWFNLA8jM.png" alt="sample1">
        <div class="overlay" style="font-size: 1vw; font-style: bold;"> Serenade in Sunlight <div class="overlay-subtext" style="font-size: 0.75em; font-style: italic;">"Caelus, immersed in music, strums his guitar in a room bathed in soft afternoon light."</div>
        </div>
      </div>
    </td>
  </tr>
</table>

<details>
  <summary>Generation Parameter</summary>
  <pre>
{
  "prompt": "cinematic photo (masterpiece), (best quality), (ultra-detailed),  caelus, honkai: star rail, 1boy, solo, playing guitar, living room, grey hair, short hair, yellow eyes, downturned eyes, passionate expression, casual clothes, acoustic guitar, sheet music stand, carpet, couch, window, sitting pose, strumming guitar, eyes closed., illustration, disheveled hair, detailed eyes, perfect composition, moist skin, intricate details, earrings . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
  "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, uglylongbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality",
  "resolution": "1216 x 832",
  "guidance_scale": 12,
  "num_inference_steps": 50,
  "seed": 1521939308,
  "sampler": "Euler a",
  "enable_lcm": false,
  "sdxl_style": "Photographic",
  "quality_tags": "Heavy",
  "refine_prompt": true,
  "use_lora": null,
  "use_upscaler": {
    "upscale_method": "nearest-exact",
    "upscaler_strength": 0.55,
    "upscale_by": 1.5,
    "new_resolution": "1824 x 1248"
  },
  "datetime": "2023-11-25 07:08:39.622020"
}
  </pre>
</details>

<table class="custom-table">
  <tr>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/eedrvT_hQjVb4rz5CmwOq.png" alt="sample1">
        <div class="overlay" style="font-size: 1vw; font-style: bold;"> Night Market Glow <div class="overlay-subtext" style="font-size: 0.75em; font-style: italic;">"Kafka serves up culinary delights, her smile as bright as the surrounding festival lights."</div>
        </div>
      </div>
    </td>
  </tr>
</table>

<details>
  <summary>Generation Parameter</summary>
  <pre>
{
  "prompt": "cinematic photo (masterpiece), (best quality), (ultra-detailed), 1girl, solo, kafka, enjoying a street food festival, dark purple hair, shoulder length, hair clip, blue eyes, upturned eyes, excited expression, casual clothes, food stalls, variety of cuisines, people, outdoor seating, string lights, standing pose, holding a plate of food, trying new dishes, laughing with friends, experiencing the vibrant food culture., illustration, disheveled hair, detailed eyes, perfect composition, moist skin, intricate details, earrings . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
  "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, uglylongbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality",
  "resolution": "1216 x 832",
  "guidance_scale": 12,
  "num_inference_steps": 50,
  "seed": 1082676886,
  "sampler": "Euler a",
  "enable_lcm": false,
  "sdxl_style": "Photographic",
  "quality_tags": "Heavy",
  "refine_prompt": false,
  "use_lora": null,
  "use_upscaler": {
    "upscale_method": "nearest-exact",
    "upscaler_strength": 0.55,
    "upscale_by": 1.5,
    "new_resolution": "1824 x 1248"
  },
  "datetime": "2023-11-25 06:51:53.961466"
}
  </pre>
</details>

### Multi Aspect Resolution

This model supports generating images at the following dimensions:
| Dimensions      | Aspect Ratio    |
|-----------------|-----------------|
| 1024 x 1024     | 1:1 Square      |
| 1152 x 896      | 9:7             |
| 896 x 1152      | 7:9             |
| 1216 x 832      | 19:13           |
| 832 x 1216      | 13:19           |
| 1344 x 768      | 7:4 Horizontal  |
| 768 x 1344      | 4:7 Vertical    |
| 1536 x 640      | 12:5 Horizontal |
| 640 x 1536      | 5:12 Vertical   |

## Examples 


## Training and Hyperparameters

- **Animagine XL** was trained on a 1x A100 GPU with 80GB memory. The training process encompassed two stages:
  - **Feature Alignment Stage**: Utilized 170k images to acquaint the model with basic anime concepts.
  - **Aesthetic Tuning Stage**: Employed 83k high-quality synthetic datasets to refine the model's art style.

### Hyperparameters

- Global Epochs: 20
- Learning Rate: 1e-6
- Batch Size: 32
- Train Text Encoder: True
- Image Resolution: 1024 (2048 x 512)
- Mixed-Precision: fp16

*Note: The model's training configuration is subject to future enhancements.*

## Model Comparison (Animagine XL 1.0 vs Animagine XL 2.0)

### Image Comparison

In the second iteration (Animagine XL 2.0), we have addressed the 'broken neck' issue prevalent in poses like "looking back" and "from behind". Now, characters are consistently "looking at viewer" by default, enhancing the naturalism and accuracy of the generated images.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/oSssetgmuLEV6RlaSC5Tr.png)

### Training Config

| Configuration Item    | Animagine XL 1.0   | Animagine XL 2.0        |
|-----------------------|--------------------|--------------------------|
| **GPU**               | A100 40G           | A100 80G                 |
| **Dataset**           | 8000 images        | 170k + 83k images        |
| **Global Epochs**     | Not Applicable     | 20                       |
| **Learning Rate**     | 4e-7               | 1e-6                     |
| **Batch Size**        | 16                 | 32                       |
| **Train Text Encoder**| False              | True                     |
| **Train Special Tags**| False              | True                     |
| **Image Resolution**  | 1024               | 1024                     |
| **Bucket Resolution** | 1024 x 256         | 2048 x 512               |
| **Caption Dropout**   | 0.5                | 0                        |

## Direct Use

The Animagine XL 2.0 model, with its advanced text-to-image diffusion capabilities, is highly versatile and can be applied in various fields:

- **Art and Design:** This model is a powerful tool for artists and designers, enabling the creation of unique and high-quality anime-style artworks. It can serve as a source of inspiration and a means to enhance creative processes.
- **Education:** In educational contexts, Animagine XL 2.0 can be used to develop engaging visual content, assisting in teaching concepts related to art, technology, and media.
- **Entertainment and Media:** The model's ability to generate detailed anime images makes it ideal for use in animation, graphic novels, and other media production, offering a new avenue for storytelling.
- **Research:** Academics and researchers can leverage Animagine XL 2.0 to explore the frontiers of AI-driven art generation, study the intricacies of generative models, and assess the model's capabilities and limitations.
- **Personal Use:** Anime enthusiasts can use Animagine XL 2.0 to bring their imaginative concepts to life, creating personalized artwork based on their favorite genres and styles.

## Limitations

The Animagine XL 2.0 model, while advanced in its capabilities, has certain limitations that users should be aware of:

- **Style Bias:** The model exhibits a bias towards a specific art style, as it was fine-tuned using approximately 80,000 images with a similar aesthetic. This may limit the diversity in the styles of generated images.
- **Rendering Challenges:** There are occasional inaccuracies in rendering hands or feet, which may not always be depicted with high fidelity.
- **Realism Constraint:** Animagine XL 2.0 is not designed for generating realistic images, given its focus on anime-style content.
- **Natural Language Limitations:** The model may not perform optimally when prompted with natural language descriptions, as it is tailored more towards anime-specific terminologies and styles.
- **Dataset Scope:** Currently, the model is primarily effective in generating content related to the 'Honkai' series and 'Genshin Impact' due to the dataset's scope. Expansion to include more diverse concepts is planned for future iterations.
- **NSFW Content Generation:** The model is not proficient in generating NSFW content, as it was not a focus during the training process, aligning with the intention to promote safe and appropriate content generation.

## Acknowledgements

We extend our gratitude to:

- **Chai AI:** For the open-source grant ([Chai AI](https://www.chai-research.com/)) supporting our research.
- **Kohya SS:** For providing the essential training script.
- **Camenduru Server Community:** For invaluable insights and support.
- **NovelAI:** For inspiring the Quality Tags feature.
- **Waifu DIffusion Team:** for inspiring the optimal training pipeline with bigger datasets.
- **Shadow Lilac:** For the image classification model ([shadowlilac/aesthetic-shadow](https://huggingface.co/shadowlilac/aesthetic-shadow)) crucial in our quality assessment process.

<h1 class="title">
  <span>Anything you can Imagine!</span>
</h1>
