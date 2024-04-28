---
license: openrail++
tags:
- text-to-image
- Pixart-α
---

<p align="center">
  <img src="asset/logo.png"  height=120>
</p>

<div style="display:flex;justify-content: center">
  <a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha"><img src="https://img.shields.io/static/v1?label=Demo&message=Huggingface&color=yellow"></a> &ensp;
  <a href="https://pixart-alpha.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2310.00426"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://colab.research.google.com/drive/1jZ5UZXk7tcpTfVwnX33dDuefNMcnW9ME?usp=sharing"><img src="https://img.shields.io/static/v1?label=Free%20Trial&message=Google%20Colab&logo=google&color=orange"></a> &ensp;
  <a href="https://github.com/orgs/PixArt-alpha/discussions"><img src="https://img.shields.io/static/v1?label=Discussion&message=Github&color=green&logo=github"></a> &ensp;
</div>

# 🐱 Pixart-α Model Card
![row01](asset/images/teaser.png)

## Model
![pipeline](asset/images/model.png)

[Pixart-α](https://arxiv.org/abs/2310.00426) consists of pure transformer blocks for latent diffusion: 
It can directly generate 1024px images from text prompts within a single sampling process.

Source code is available at https://github.com/PixArt-alpha/PixArt-alpha.

### Model Description

- **Developed by:** Pixart-α
- **Model type:** Diffusion-Transformer-based text-to-image generative model
- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. 
It is a [Transformer Latent Diffusion Model](https://arxiv.org/abs/2310.00426) that uses one fixed, pretrained text encoders ([T5](
https://huggingface.co/DeepFloyd/t5-v1_1-xxl))
and one latent feature encoder ([VAE](https://arxiv.org/abs/2112.10752)).
- **Resources for more information:** Check out our [GitHub Repository](https://github.com/PixArt-alpha/PixArt-alpha) and the [Pixart-α report on arXiv](https://arxiv.org/abs/2310.00426).

### Model Sources

For research purposes, we recommend our `generative-models` Github repository (https://github.com/PixArt-alpha/PixArt-alpha), 
which is more suitable for both training and inference and for which most advanced diffusion sampler like [SA-Solver](https://arxiv.org/abs/2309.05019) will be added over time.
[Hugging Face](https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha) provides free Pixart-α inference.
- **Repository:** https://github.com/PixArt-alpha/PixArt-alpha
- **Demo:** https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha

# 🔥🔥🔥 Why PixArt-α? 
## Training Efficiency
PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%.
![Training Efficiency.](asset/images/efficiency.svg)

| Method    | Type | #Params | #Images | A100 GPU days |
|-----------|------|---------|---------|---------------|
| DALL·E    | Diff | 12.0B   | 1.54B   |               |
| GLIDE     | Diff | 5.0B    | 5.94B   |               |
| LDM       | Diff | 1.4B    | 0.27B   |               |
| DALL·E 2  | Diff | 6.5B    | 5.63B   | 41,66         |
| SDv1.5    | Diff | 0.9B    | 3.16B   | 6,250         |
| GigaGAN   | GAN  | 0.9B    | 0.98B   | 4,783         |
| Imagen    | Diff | 3.0B    | 15.36B  | 7,132         |
| RAPHAEL   | Diff | 3.0B    | 5.0B    | 60,000        |
| PixArt-α  | Diff | 0.6B    | 0.025B  | 675           |


## Evaluation
![comparison](asset/images/user-study.png)
The chart above evaluates user preference for Pixart-α over SDXL 0.9, Stable Diffusion 2, DALLE-2 and DeepFloyd. 
The Pixart-α base model performs comparable or even better than the existing state-of-the-art models.



### 🧨 Diffusers 

Make sure to upgrade diffusers to >= 0.22.0:
```
pip install -U diffusers --upgrade
```

In addition make sure to install `transformers`, `safetensors`, `sentencepiece`, and `accelerate`:
```
pip install transformers accelerate safetensors sentencepiece
```

To just use the base model, you can run:


```py
from diffusers import PixArtAlphaPipeline
import torch

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"
images = pipe(prompt=prompt).images[0]
```

When using `torch >= 2.0`, you can improve the inference speed by 20-30% with torch.compile. Simple wrap the unet with torch compile before running the pipeline:
```py
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
```

If you are limited by GPU VRAM, you can enable *cpu offloading* by calling `pipe.enable_model_cpu_offload`
instead of `.to("cuda")`:

```diff
- pipe.to("cuda")
+ pipe.enable_model_cpu_offload()
```

For more information on how to use Pixart-α with `diffusers`, please have a look at [the Pixart-α Docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart).

### Free Google Colab
You can use Google Colab to generate images from PixArt-α free of charge. Click [here](https://colab.research.google.com/drive/1jZ5UZXk7tcpTfVwnX33dDuefNMcnW9ME?usp=sharing) to try.

## Uses

### Direct Use

The model is intended for research purposes only. Possible research areas and tasks include

- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.

- Probing and understanding the limitations and biases of generative models.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Limitations and Bias

### Limitations


- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- fingers, .etc in general may not be generated properly.
- The autoencoding part of the model is lossy.

### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.
