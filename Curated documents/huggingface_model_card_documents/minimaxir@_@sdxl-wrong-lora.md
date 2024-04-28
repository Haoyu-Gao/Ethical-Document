---
license: mit
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
base_model: stabilityai/stable-diffusion-xl-base-1.0
inference: true
---

# sdxl-wrong-lora

![](img/header.webp)

A LoRA for SDXL 1.0 Base which improves output image quality after loading it and using `wrong` as a negative prompt during inference. You can demo image generation using this LoRA in [this Colab Notebook](https://colab.research.google.com/github/minimaxir/sdxl-experiments/blob/main/sdxl_image_generation.ipynb).

The LoRA is also available in a `safetensors` format for other UIs such as A1111; however this LoRA was created using `diffusers` and I cannot guarantee its efficacy outside of it.

Benefits of using this LoRA:

- Higher detail in textures/fabrics, particularly at full 1024x1024 resolution.
- Higher color saturation and vibrance.
- Higher sharpness for blurry/background objects.
- Better at anatomically-correct hands.
- Less likely to have random artifacts.
- Appears to allow the model to follow the input prompt with a more expected behavior, particularly with prompt weighting such as the [Compel](https://github.com/damian0815/compel) syntax.

## Usage

The LoRA can be loaded using `load_lora_weights` like any other LoRA in `diffusers`:

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

base.load_lora_weights("minimaxir/sdxl-wrong-lora")

_ = base.to("cuda")
```

During image generation, use `wrong` as the negative prompt. That's it!

## Examples

**Left image** is the base model output (no LoRA) + refiner, **right image** is base (w/ LoRA) + refiner + `wrong` negative prompt. Both generations use the same seed.

I have also [released a Colab Notebook](https://colab.research.google.com/github/minimaxir/sdxl-experiments/blob/main/sdxl_wrong_comparison.ipynb) to generate these kinds of side-by-side comparison images, although the seeds listed will not give the same results since they were generated on a different GPU/CUDA than the Colab Notebook.

`realistic human Shrek blogging at a computer workstation, hyperrealistic award-winning photo for vanity fair` (cfg = 13, seed = 56583700)

![](img/example1.webp)

`pepperoni pizza in the shape of a heart, hyperrealistic award-winning professional food photography` (cfg = 13, seed = 75789081)

![](img/example2.webp)

`presidential painting of realistic human Spongebob Squarepants wearing a suit, (oil on canvas)+++++` (cfg = 13, seed = 85588026)

![](img/example3.webp)

`San Francisco panorama attacked by (one massive kitten)++++, hyperrealistic award-winning photo by the Associated Press` (cfg = 13, seed = 45454868)

![](img/example4.webp)

`hyperrealistic death metal album cover featuring edgy moody realistic (human Super Mario)++, edgy and moody` (cfg = 13, seed = 30416580)

![](img/example5.webp)

## Methodology

The methodology and motivation for creating this LoRA is similar to my [wrong SD 2.0 textual inversion embedding](https://huggingface.co/minimaxir/wrong_embedding_sd_2_0) by training on a balanced variety of undesirable outputs, except trained as a LoRA since textual inversion with SDXL is complicated. The base images were generated from SDXL itself, with some prompt weighting to emphasize undesirable attributes for test images.

You can see the code to generate the wrong images [in this Jupyter Notebook](https://github.com/minimaxir/sdxl-experiments/blob/main/wrong_image_generator.ipynb).

## Notes

- The intuitive way to think about how this LoRA works is that on training start, it indicates an undesirable area of the vast highdimensional latent space which the rest of the diffusion process will move away from. This may work more effectively than textual inversion but more testing needs to be done.
- The description of this LoRA is very careful to not state that the output is objectively _better_ than not using LoRA, because everything is subjective and there are use cases where vibrant output is not desired. For most use cases, the output should be better desired however.
- It's possible to use `not wrong` in the normal prompt itself but in testing it has not much effect.
- You can use other negative prompts in conjunction with the `wrong` prompt but you may want to weight them appropriately.
- All the Notebooks noted here are available [in this GitHub repo](https://github.com/minimaxir/sdxl-experiments).
