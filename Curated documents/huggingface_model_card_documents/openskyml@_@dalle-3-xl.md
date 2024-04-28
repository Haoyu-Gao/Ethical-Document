---
language:
- en
- fr
- ru
license: mit
library_name: diffusers
tags:
- text-to-image
- stable-diffusion
- lora
- dalle-3
- dalle
- deepvision
- diffusers
- template:sd-lora
- openskyml
widget:
- text: a close up of a fire breathing pokemon figure, digital art, trending on polycount,
    real life charmander, sparks flying, photo-realistic unreal engine, pokemon in
    the wild
  output:
    url: images/00002441-10291230.jpeg
- text: astronaut riding a llama on Mars
  output:
    url: images/c96a4147-b14d-4e71-8c08-e04c31c8be18.jpg
- text: cube cutout of an isometric programmer bedroom, 3d art, muted colors, soft
    lighting, high detail, concept art, behance, ray tracing
  output:
    url: images/b7ad0f38-5d2a-48cd-b7d4-b94be1d23c40.jpg
- text: mario, mario (series), 1boy, blue overalls, brown hair, facial hair, gloves,
    hat, male focus, mustache, overalls, red headwear, red shirt, shirt, short hair,
    upper body, white gloves.
  parameters:
    negative_prompt: (worst quality, low quality, normal quality, lowres, low details,
      oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad
      photo, bad photography, bad art:1.4), (watermark, signature, text font, username,
      error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur,
      blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly
      lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts,
      out of focus, glitch, duplicate, (bad hands, bad anatomy, bad body, bad face,
      bad teeth, bad arms, bad legs, deformities:1.3)
  output:
    url: images/00002489-10291327.jpeg
base_model: stablediffusionapi/juggernaut-xl-v5
instance_prompt: <lora:Dall-e_3_0.3-v2-000003>
pipeline_tag: text-to-image
---
# DALL•E 3 XL

<Gallery />

## Model description 

This is a test model very similar to Dall•E 3.

## Official demo

You can use official demo on Spaces: [try](https://huggingface.co/spaces/openskyml/dalle-3).

### Published on HF.co with the OpenSkyML team


## Download model

Weights for this model are available in Safetensors format.

[Download](/openskyml/dalle-3/tree/main) them in the Files & versions tab.