---
language:
- en
license: creativeml-openrail-m
library_name: diffusers
tags:
- art
- anime
- stable diffusion
pipeline_tag: text-to-image
---

MeinaMix Objective is to be able to do good art with little prompting.

For examples and prompts, please checkout: https://civitai.com/models/7240/meinamix
I have a discord server where you can post images that you generated, discuss prompt and/or ask for help.

https://discord.gg/XC9nGZNDUd If you like one of my models and want to support their updates

I've made a ko-fi page; https://ko-fi.com/meina where you can pay me a coffee <3

And a Patreon page; https://www.patreon.com/MeinaMix where you can support me and get acess to beta of my models!

You may also try this model using Sinkin.ai: https://sinkin.ai/m/vln8Nwr

MeinaMix and the other of Meinas will ALWAYS be FREE.

Recommendations of use: Enable Quantization in K samplers.

Hires.fix is needed for prompts where the character is far away in order to make decent images, it drastically improve the quality of face and eyes!

Recommended parameters:

Sampler: Euler a: 40 to 60 steps.

Sampler: DPM++ SDE Karras: 20 to 30 steps.

Sampler: DPM++ 2M Karras: 20 to 40 steps.

CFG Scale: 7.

Resolutions: 512x768, 512x1024 for Portrait!

Resolutions: 768x512, 1024x512, 1536x512 for Landscape!

Hires.fix: R-ESRGAN 4x+Anime6b, with 10 steps at 0.3 up to 0.5 denoising.

Clip Skip: 2.

Negatives: ' (worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic) '