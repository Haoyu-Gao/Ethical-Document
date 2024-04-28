---
pipeline_tag: text-to-image
inference: false
---

# SD-Turbo Model Card

<!-- Provide a quick summary of what the model is/does. -->
![row01](output_tile.jpg)
SD-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation.
We release SD-Turbo as a research artifact, and to study small, distilled text-to-image models. For increased quality and prompt understanding, 
we recommend [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo/).

## Model Details

### Model Description
SD-Turbo is a distilled version of [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1), trained for real-time synthesis. 
SD-Turbo is based on a novel training method called Adversarial Diffusion Distillation (ADD) (see the [technical report](https://stability.ai/research/adversarial-diffusion-distillation)), which allows sampling large-scale foundational 
image diffusion models in 1 to 4 steps at high image quality. 
This approach uses score distillation to leverage large-scale off-the-shelf image diffusion models as a teacher signal and combines this with an
adversarial loss to ensure high image fidelity even in the low-step regime of one or two sampling steps. 

- **Developed by:** Stability AI
- **Funded by:** Stability AI
- **Model type:** Generative text-to-image model
- **Finetuned from model:** [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

### Model Sources

For research purposes, we recommend our `generative-models` Github repository (https://github.com/Stability-AI/generative-models), 
which implements the most popular diffusion frameworks (both training and inference).

- **Repository:** https://github.com/Stability-AI/generative-models
- **Paper:** https://stability.ai/research/adversarial-diffusion-distillation
- **Demo [for the bigger SDXL-Turbo]:** http://clipdrop.co/stable-diffusion-turbo


## Evaluation
![comparison1](image_quality_one_step.png)
![comparison2](prompt_alignment_one_step.png)
The charts above evaluate user preference for SD-Turbo over other single- and multi-step models.
SD-Turbo evaluated at a single step is preferred by human voters in terms of image quality and prompt following over LCM-Lora XL and LCM-Lora 1.5.

**Note:** For increased quality, we recommend the bigger version [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo/).
For details on the user study, we refer to the [research paper](https://stability.ai/research/adversarial-diffusion-distillation).


## Uses

### Direct Use

The model is intended for research purposes only. Possible research areas and tasks include

- Research on generative models.
- Research on real-time applications of generative models.
- Research on the impact of real-time generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.

Excluded uses are described below.

### Diffusers

```
pip install diffusers transformers accelerate --upgrade
```

- **Text-to-image**:

SD-Turbo does not make use of `guidance_scale` or `negative_prompt`, we disable it with `guidance_scale=0.0`.
Preferably, the model generates images of size 512x512 but higher image sizes work as well.
A **single step** is enough to generate high quality images.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
```

- **Image-to-image**:

When using SD-Turbo for image-to-image generation, make sure that `num_inference_steps` * `strength` is larger or equal 
to 1. The image-to-image pipeline will run for `int(num_inference_steps * strength)` steps, *e.g.* 0.5 * 2.0 = 1 step in our example 
below.

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
```

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, 
and therefore using the model to generate such content is out-of-scope for the abilities of this model.
The model should not be used in any way that violates Stability AI's [Acceptable Use Policy](https://stability.ai/use-policy).

## Limitations and Bias

### Limitations
- The quality and prompt alignment is lower than that of [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo/).
- The generated images are of a fixed resolution (512x512 pix), and the model does not achieve perfect photorealism.
- The model cannot render legible text.
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.


### Recommendations

The model is intended for research purposes only.

## How to Get Started with the Model

Check out https://github.com/Stability-AI/generative-models
