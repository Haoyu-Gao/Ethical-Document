---
license: apache-2.0
pipeline_tag: text-to-image
inference: false
---
# Kandinsky-3: Text-to-image Diffusion Model

![](assets/title.jpg)

[Post](https://habr.com/ru/companies/sberbank/articles/775590/) | [Generate](https://fusionbrain.ai) | [Telegram-bot](https://t.me/kandinsky21_bot) | [Report]

## Description:

Kandinsky 3.0 is an open-source text-to-image diffusion model built upon the Kandinsky2-x model family. In comparison to its predecessors, Kandinsky 3.0 incorporates more data and specifically related to Russian culture, which allows to generate pictures related to Russin culture. Furthermore, enhancements have been made to the text understanding and visual quality of the model, achieved by increasing the size of the text encoder and Diffusion U-Net models, respectively.

For more information: details of training, example of generations check out our [post](https://habr.com/ru/companies/sberbank/articles/775590/). The english version will be released in a couple of days.

## Architecture details:


![](assets/kandinsky.jpg)


Architecture consists of three parts:

+ Text encoder Flan-UL2 (encoder part) - 8.6B
+ Latent Diffusion U-Net - 3B
+ MoVQ encoder/decoder - 267M


## Models

We release our two models:

+ Base: Base text-to-image diffusion model. This model was trained over 2M steps on 400 A100
+ Inpainting: Inpainting version of the model. The model was initialized from final checkpoint of base model and trained 250k steps on 300 A100.

## Installing

Make sure to install `diffusers` from main as well as Transformers, Accelerate

```
pip install git+https://github.com/huggingface/diffusers.git
pip install --upgrade transformers accelerate
```

## How to use:

TODO

### Text-2-Image

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
        
prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
```

### Image-2-Image

```python
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
        
prompt = "A painting of the inside of a subway train with tiny raccoons."
image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png")

generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(prompt, image=image, strength=0.75, num_inference_steps=25, generator=generator).images[0]
```

## Examples of generations

<hr>

<table class="center">
<tr>
  <td><img src="assets/photo_8.jpg" raw=true></td>
  <td><img src="assets/photo_15.jpg"></td>
  <td><img src="assets/photo_16.jpg"></td>
  <td><img src="assets/photo_17.jpg"></td>
</tr>
<tr>
  <td width=25% align="center">"A beautiful landscape outdoors scene in the crochet knitting art style, drawing in style by Alfons Mucha"</td>
  <td width=25% align="center">"gorgeous phoenix, cosmic, darkness, epic, cinematic, moonlight, stars, high - definition, texture,Oscar-Claude Monet"</td>
  <td width=25% align="center">"a yellow house at the edge of the danish fjord, in the style of eiko ojala, ingrid baars, ad posters, mountainous vistas, george ault, realistic details, dark white and dark gray, 4k"</td>
  <td width=25% align="center">"dragon fruit head, upper body, realistic, illustration by Joshua Hoffine Norman Rockwell, scary, creepy, biohacking, futurism, Zaha Hadid style"</td>
</tr>
<tr>
  <td><img src="assets/photo_2.jpg" raw=true></td>
  <td><img src="assets/photo_19.jpg"></td>
  <td><img src="assets/photo_13.jpg"></td>
  <td><img src="assets/photo_14.jpg"></td>
</tr>
<tr>
  <td width=25% align="center">"Amazing playful nice cute strawberry character, dynamic poze, surreal fantazy garden background, gorgeous masterpice, award winning photo, soft natural lighting, 3d, Blender, Octane render, tilt - shift, deep field, colorful, I can't believe how beautiful this is, colorful, cute and sweet baby - loved photo"</td>
  <td width=25% align="center">"beautiful fairy-tale desert, in the sky a wave of sand merges with the milky way, stars, cosmism, digital art, 8k"</td>
  <td width=25% align="center">"Car, mustang, movie, person, poster, car cover, person, in the style of alessandro gottardo, gold and cyan, gerald harvey jones, reflections, highly detailed illustrations, industrial urban scenes""</td>
  <td width=25% align="center">"cloud in blue sky, a red lip, collage art, shuji terayama, dreamy objects, surreal, criterion collection, showa era, intricate details, mirror"</td>
</tr>

</table>

<hr>

## Authors

+ Vladimir Arkhipkin: [Github](https://github.com/oriBetelgeuse)
+ Anastasia Maltseva [Github](https://github.com/NastyaMittseva)
+ Andrei Filatov [Github](https://github.com/anvilarth), 
+ Igor Pavlov: [Github](https://github.com/boomb0om)
+ Julia Agafonova 
+ Arseniy Shakhmatov: [Github](https://github.com/cene555), [Blog](https://t.me/gradientdip)
+ Andrey Kuznetsov: [Github](https://github.com/kuznetsoffandrey), [Blog](https://t.me/complete_ai)
+ Denis Dimitrov: [Github](https://github.com/denndimitrov), [Blog](https://t.me/dendi_math_ai)