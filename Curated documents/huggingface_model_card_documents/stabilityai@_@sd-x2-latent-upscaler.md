---
license: openrail++
tags:
- stable-diffusion
inference: false
---

# Stable Diffusion x2 latent upscaler model card

This model card focuses on the latent diffusion-based upscaler developed by [Katherine Crowson](https://github.com/crowsonkb/k-diffusion) 
in collaboration with [Stability AI](https://stability.ai/). 
This model was trained on a high-resolution subset of the LAION-2B dataset. 
It is a diffusion model that operates in the same latent space as the Stable Diffusion model, which is decoded into a full-resolution image. 
To use it with Stable Diffusion, You can take the generated latent from Stable Diffusion and pass it into the upscaler before decoding with your standard VAE. 
Or you can take any image, encode it into the latent space, use the upscaler, and decode it. 

**Note**: 
This upscaling model is designed explicitely for **Stable Diffusion** as it can upscale Stable Diffusion's latent denoised image embeddings.
This allows for very fast text-to-image + upscaling pipelines as all intermeditate states can be kept on GPU. More for information, see example below.
This model works on all [Stable Diffusion checkpoints](https://huggingface.co/models?other=stable-diffusion)

| ![upscaler.jpg](https://pbs.twimg.com/media/FhK0YjAVUAUtBbx?format=jpg&name=4096x4096) |
|:--:|
Image by Tanishq Abraham from [Stability AI](https://stability.ai/) originating from [this tweet](https://twitter.com/StabilityAI/status/1590531958815064065)|

Original output image             |  2x upscaled output image
:-------------------------:|:-------------------------:
![](https://pbs.twimg.com/media/Fg8UijAaEAAqfvS?format=png&name=small)  |  ![](https://pbs.twimg.com/media/Fg8UjCmaMAAAUdS?format=jpg&name=medium)

- Use it with 🧨 [`diffusers`](https://huggingface.co/stabilityai/sd-x2-latent-upscaler#examples)


## Model Details
- **Developed by:** Katherine Crowson
- **Model type:** Diffusion-based latent upscaler
- **Language(s):** English
- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)


## Examples

Using the [🤗's Diffusers library](https://github.com/huggingface/diffusers) to run latent upscaler on top of any `StableDiffusionUpscalePipeline` checkpoint 
to enhance its output image resolution by a factor of 2.

```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate scipy safetensors
```

```python
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline.to("cuda")

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
upscaler.to("cuda")

prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
generator = torch.manual_seed(33)

# we stay in latent space! Let's make sure that Stable Diffusion returns the image
# in latent space
low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]

# Let's save the upscaled image under "upscaled_astronaut.png"
upscaled_image.save("astronaut_1024.png")

# as a comparison: Let's also save the low-res image
with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]

image.save("astronaut_512.png")
```

**Result**:

*512-res Astronaut*
![ow_res](./astronaut_512.png)

*1024-res Astronaut*
![upscaled](./astronaut_1024.png)

**Notes**:
- Despite not being a dependency, we highly recommend you to install [xformers](https://github.com/facebookresearch/xformers) for memory efficient attention (better performance)
- If you have low GPU RAM available, make sure to add a `pipe.enable_attention_slicing()` after sending it to `cuda` for less VRAM usage (to the cost of speed)

# Uses

## Direct Use 
The model is intended for research purposes only. Possible research areas and tasks include

- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.

Excluded uses are described below.

 ### Misuse, Malicious Use, and Out-of-Scope Use
_Note: This section is originally taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), was used for Stable Diffusion v1, but applies in the same way to Stable Diffusion v2_.

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

#### Out-of-Scope Use
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

#### Misuse and Malicious Use
Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

## Limitations and Bias

### Limitations

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The model was trained mainly with English captions and will not work as well in other languages.
- The autoencoding part of the model is lossy
- The model was trained on a subset of the large-scale dataset
  [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, we have filtered the dataset using LAION's NFSW detector (see Training section).

### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. 
Stable Diffusion vw was primarily trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), 
which consists of images that are limited to English descriptions. 
Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. 
This affects the overall output of the model, as white and western cultures are often set as the default. Further, the 
ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.
Stable Diffusion v2 mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.