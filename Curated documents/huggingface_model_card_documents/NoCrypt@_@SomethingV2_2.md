---
language:
- en
license: creativeml-openrail-m
library_name: diffusers
tags:
- stable-diffusion
- text-to-image
- safetensors
- diffusers
thumbnail: https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/thumbnail.webp
inference: true
widget:
- text: masterpiece, masterpiece, masterpiece, best quality, ultra-detailed, 1girl,
    hatsune miku, blue hair, upper body, looking at viewer, ?, negative space, bioluminescence,
    bioluminescence, bioluminescence, darkness, wind, butterfly, black background,
    portrait, ice
  example_title: example
---

<center>

<img src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/Artboard%201.png"/>


<h1 style="font-size:1.6rem;">
  <b>
    SomethingV2.2
  </b>
</h1>

<p>
  Welcome to SomethingV2.2 - an improved anime latent diffusion model from <a href="https://huggingface.co/NoCrypt/SomethingV2">SomethingV2</a>

  A lot of things are being discovered lately, such as a way to merge model using mbw automatically, offset noise to get much darker result, and even VAE tuning. This model is intended to use all of those features as the improvements, here's some improvements that have been made:
</p>

<img src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/Artboard%202.png"/>


<h2>Can't trust the numbers? Here's some proof</h2>

</center>

![](https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/xyz_grid-0000-3452449180-masterpiece%2C%20best%20quality%2C%20ultra-detailed%2C%202girls%2C%20upper%20body%2C%20looking%20at%20viewer%2C%20_%2C%20negative%20space%2C%20(bioluminescence_1.2)%2C%20dark.png)
![](https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/xyz_grid-0003-72332473-masterpiece%2C%20best%20quality%2C%20hatsune%20miku%2C%20white%20shirt%2C%20darkness%2C%20dark%20background.png)


<img style="display:inline;margin:0;padding:0;" src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/00019-1829045217-masterpiece%2C%20best%20quality%2C%20hatsune%20miku%2C%201girl%2C%20white%20shirt%2C%20blue%20necktie%2C%20bare%20shoulders%2C%20very%20detailed%20background%2C%20hands%20on%20ow.png" width="32%"/>
<img style="display:inline;margin:0;padding:0;" src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/00018-1769428138-masterpiece%2C%20best%20quality%2C%20hatsune%20miku%2C%201girl%2C%20white%20shirt%2C%20blue%20necktie%2C%20bare%20shoulders%2C%20very%20detailed%20background%2C%20hands%20on%20ow.png" width="32%"/>
<img style="display:inline;margin:0;padding:0;" src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/00020-3514023396-masterpiece%2C%20best%20quality%2C%20hatsune%20miku%2C%201girl%2C%20white%20shirt%2C%20blue%20necktie%2C%20bare%20shoulders%2C%20very%20detailed%20background%2C%20cafe%2C%20angry.png" width="32%"/>


<details><summary><big><b>Prompts</b></big></summary>

```yaml
masterpiece, best quality, ultra-detailed, 2girls, upper body, looking at viewer, ?, negative space, (bioluminescence:1.2), darkness, wind, butterfly, black background, glowing,
AND masterpiece, best quality, ultra-detailed, 2girls, hatsune miku, upper body, looking at viewer, ?, negative space, (bioluminescence:1.2), darkness, wind, butterfly, black background, glowing, (blue theme:1.2)
AND masterpiece, best quality, ultra-detailed, 2girls, hakurei reimu, (brown hair:1.1), upper body, looking at viewer, ?, negative space, (bioluminescence:1.2), darkness, wind, butterfly, black background, glowing, (red theme:1.2)
Negative prompt: EasyNegative
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 3452449180, Size: 816x504, Model: somethingv2_1, Denoising strength: 0.58, Clip skip: 2, ENSD: 31337, Latent Couple: "divisions=1:1,1:2,1:2 positions=0:0,0:0,0:1 weights=0.2,0.8,0.8 end at step=13", Hires upscale: 1.9, Hires steps: 12, Hires upscaler: Latent (nearest-exact)
```

```yaml
masterpiece, best quality, hatsune miku, white shirt, darkness, dark background
Negative prompt: EasyNegative
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 72332473, Size: 504x600, Model: somethingv2_1, Denoising strength: 0.58, Clip skip: 2, ENSD: 31337, Hires upscale: 1.85, Hires steps: 12, Hires upscaler: Latent (nearest-exact)
```

```yaml
masterpiece, best quality, hatsune miku, 1girl, white shirt, blue necktie, bare shoulders, very detailed background, hands on own cheeks, open mouth, one eye closed, clenched teeth, smile
Negative prompt: EasyNegative, tattoo, (shoulder tattoo:1.0), (number tattoo:1.3), frills
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1829045217, Size: 456x592, Model: SomethingV2_2, Denoising strength: 0.53, Clip skip: 2, ENSD: 31337, Hires upscale: 1.65, Hires steps: 12, Hires upscaler: Latent (nearest-exact), Discard penultimate sigma: True
```

```yaml
masterpiece, best quality, hatsune miku, 1girl, white shirt, blue necktie, bare shoulders, very detailed background, hands on own cheeks, open mouth, eyez closed, clenched teeth, smile, arms behind back,
Negative prompt: EasyNegative, tattoo, (shoulder tattoo:1.0), (number tattoo:1.3), frills
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1769428138, Size: 456x592, Model: SomethingV2_2, Denoising strength: 0.53, Clip skip: 2, ENSD: 31337, Hires upscale: 1.65, Hires steps: 12, Hires upscaler: Latent (nearest-exact), Discard penultimate sigma: True
```

```yaml
masterpiece, best quality, hatsune miku, 1girl, white shirt, blue necktie, bare shoulders, very detailed background, cafe, angry, crossed arms, detached sleeves, light particles,
Negative prompt: EasyNegative, tattoo, (shoulder tattoo:1.0), (number tattoo:1.3), frills
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 3514023396, Size: 456x592, Model: SomethingV2_2, Denoising strength: 0.53, Clip skip: 2, ENSD: 31337, Hires upscale: 1.65, Hires steps: 12, Hires upscaler: Latent (nearest-exact), Discard penultimate sigma: True

```

</details>


## Non-miku examples

<img style="display:inline;margin:0;padding:0;"  width="49%" src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/00021-4018636341-masterpiece%2C%20best%20quality%2C%201girl%2C%20aqua%20eyes%2C%20baseball%20cap%2C%20blonde%20hair%2C%20closed%20mouth%2C%20earrings%2C%20green%20background%2C%20hat%2C%20hoop%20earr.png"/>
<img style="display:inline;margin:0;padding:0;"  width="49%" src="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/images/00022-1334620477-masterpiece%2C%20best%20quality%2C%20landscape.png"/>

<details><summary><big><b>Prompts</b></big></summary>

```yaml
masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt
Negative prompt: EasyNegative
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 4018636341, Size: 440x592, Model: SomethingV2_2, Denoising strength: 0.53, Clip skip: 2, ENSD: 31337, Hires upscale: 1.65, Hires steps: 13, Hires upscaler: Latent (nearest-exact)
```

```yaml
masterpiece, best quality, landscape
Negative prompt: EasyNegative
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1334620477, Size: 440x592, Model: SomethingV2_2, Denoising strength: 0.53, Clip skip: 2, ENSD: 31337, Hires upscale: 1.65, Hires steps: 13, Hires upscaler: Latent (nearest-exact)
```

</details>


## Recommended settings
- VAE: None (Baked in model, [blessed2](https://huggingface.co/NoCrypt/blessed_vae/blob/main/blessed2.vae.pt))
- Clip Skip: 2
- Sampler: DPM++ 2M Karras
- CFG Scale: 7 ± 5
- Recommended Positive Prompt: masterpiece, best quality, negative space, (bioluminescence:1.2), darkness, dark background
- Recommended Negative Prompt: [EasyNegative](https://huggingface.co/datasets/gsdf/EasyNegative)
- For better results, using hires fix is a must. 
- Hires upscaler: Latent (any variant, such as nearest-exact)


## Recipe
*Due to [SD-Silicon's Terms of use](https://huggingface.co/Xynon/SD-Silicon#terms-of-use). I must specify how the model was made*
|Model A | Model B | Interpolation Method | Weight | Name |
|---|---|---|---|---|
|[dpepmkmp](https://huggingface.co/closertodeath/dpepmkmp/blob/main/dpepmkmp.safetensors)|[silicon29-dark](https://huggingface.co/Xynon/SD-Silicon/blob/main/Silicon29/Silicon29-dark.safetensors)|MBW|Reverse Cosine|[dpepsili](https://huggingface.co/un1xx/model_dump/blob/main/bw-merge-dpepmkmp-Silicon29-dark-0.ckpt)|
|[somethingV2_1](https://huggingface.co/NoCrypt/SomethingV2/blob/main/somethingv2_1.safetensors)|[dpepsili](https://huggingface.co/un1xx/model_dump/blob/main/bw-merge-dpepmkmp-Silicon29-dark-0.ckpt)|MBW|Cosine|SomethingV2_2 raw|
|SomethingV2_2 raw|[Blessed2 VAE](https://huggingface.co/NoCrypt/blessed_vae/blob/main/blessed2.vae.pt)|Bake VAE|-|**[SomethingV2_2](https://huggingface.co/NoCrypt/SomethingV2_2/blob/main/SomethingV2_2.safetensors)**|
 

## Why not call it SomethingV4?
Since this model was based on SomethingV2 and there's not THAT much of improvements in some condition. Calling it V4 is just not right at the moment 😅

I am NoCrypt