---
license: creativeml-openrail-m
library_name: diffusers
tags:
- art
- anime
pipeline_tag: text-to-image
---

# Loli Diffusion
The goal of this project is to improve generation of loli characters since most of other models are not good at it. \
__Support me: https://www.buymeacoffee.com/jilek772003__ \
\
__Some of the models can be used online on these plarforms:__ \
__Aipictors (Japanese) - https://www.aipictors.com__ \
__Yodayo (English) - https://www.aipictors.com (comming soon with more content here)__

## Usage
It is recommende to use standard resolution such as 512x768 and EasyNegative embedding with these models. \
Positive prompt example: 1girl, solo, loli, masterpiece \
Negative prompt example: EasyNegative, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, multiple panels, aged up, old \
All examples were generated using custom workflow in ComfyUI and weren't edited using inpainting. You can load the workflow by either importing the example images or importing the workflow directly

## Useful links
Reddit: https://www.reddit.com/r/loliDiffusion \
Discord: https://discord.gg/mZ3eGeNX7S

## About
v0.4.3 \
Fixed color issue \
General improvements \
\
v0.5.3 \
Integrated VAE\
File size reduced \
CLIP force reset fix \
\
v0.6.3 \
Style improvements \
Added PastelMix and Counterfeit style \
\
v0.7.x \
Style impovements \
Composition improvements \
\
v0.8.x \
Major improvement on higher resolutions \
Style improvements \
Flexibility and responsivity \
Added support for Night Sky YOZORA model \
\
v0.9.x \
Different approach at merging, you might find v0.8.x versions better \
Changes at supported models \
\
v2.1.X EXPERIMENTAL RELEASE \
Stable Diffusion 2.1-768 based \
Default negative prompt: (low quality, worst quality:1.4), (bad anatomy), extra finger, fewer digits, jpeg artifacts \
For positive prompt it's good to include tags: anime, (masterpiece, best quality) alternatively you may achieve positive response with: (exceptional, best aesthetic, new, newest, best quality, masterpiece, extremely detailed, anime, waifu:1.2) \
Though it's Loli Diffusion model it's quite general purpose \
The ability to generate realistic images as Waifu Diffusion can was intentionally decreased \
This model performs better at higher resolutions like 768\*X or 896\*X \
\
v0.10.x \
Different approach at merging \
Better hands \
Better style inheritance \
Some changes in supported models 
\
v0.11.x \
Slight changes \
Some changes in supported models \
\
v0.13.x \
Slight model stability improvements \
Prompting loli requires lower weight now

## Examples
### YOZORA
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00597_.png"></img>
### 10th Heaven
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00632_.png"></img>
### AOM2 SFW
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00667_.png"></img>
### BASED
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00744_.png"></img>
### Counterfeit
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00849_.png"></img>
### EstheticRetroAnime
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00905_.png"></img>
### Hassaku
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00954_.png"></img>
### Koji
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_00975_.png"></img>
### Animelike
<img src="https://huggingface.co/JosefJilek/loliDiffusion/resolve/main/examples/ComfyUI_01185_.png"></img>

## Resources
https://huggingface.co/datasets/gsdf/EasyNegative \
https://huggingface.co/WarriorMama777/OrangeMixs \
https://huggingface.co/hakurei/waifu-diffusion-v1-4 \
https://huggingface.co/gsdf/Counterfeit-V2.5 \
https://civitai.com/models/12262?modelVersionId=14459 \
https://civitai.com/models/149664/based67 \
https://huggingface.co/gsdf/Counterfeit-V2.5 \
https://huggingface.co/Yntec/EstheticRetroAnime \
https://huggingface.co/dwarfbum/Hassaku \
https://huggingface.co/stb/animelike2d