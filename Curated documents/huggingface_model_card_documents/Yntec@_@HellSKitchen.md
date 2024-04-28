---
language:
- en
license: creativeml-openrail-m
library_name: diffusers
tags:
- Anime
- Style
- 2D
- Base Model
- iamxenos
- Barons
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
pipeline_tag: text-to-image
inference: true
---

# Hell's Kitchen

Kitsch-In-Sync v2 with HELLmix's compositions. Despite the Anime tag, this is a general purpose model that also does anime.

Sample and prompt:

![Sample](https://cdn-uploads.huggingface.co/production/uploads/63239b8370edc53f51cd5d42/ISTLHrwbE5v-Ai9RXEL3o.png)

Father with little daughter. A pretty cute girl sitting with Santa Claus holding Coca Cola, Christmas Theme Art by Gil_Elvgren and Haddon_Sundblom

# HELL Cola

The Coca Cola LoRA merged into HELLmix (it does not have a VAE). It's a very nice model but does not have as much creativity as Hell's Kitchen.

Sample:

![Another sample](https://cdn-uploads.huggingface.co/production/uploads/63239b8370edc53f51cd5d42/HbK8-0XyRlosEAMR5ivkx.png)

# Recipe:

- SuperMerger Weight sum Train Difference Use MBW 1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0

Model A: 

Kitsch-In-Sync v2

Model B:

HELLmix

Output Model:

HELLSKitchen

- Merge LoRA into checkpoint:

Model A: 

HELLmix

LoRA:

Coca Cola

Output Model:

HELLCola

Original pages:

https://civitai.com/models/21493/hellmix?modelVersionId=25632 (HELLmix)

https://civitai.com/models/142552?modelVersionId=163068 (Kitsch-In-Sync v2)

https://civitai.com/models/186251/coca-cola-gil-elvgrenhaddon-sundblom-pinup-style