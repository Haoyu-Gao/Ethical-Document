---
license: openrail++
tags:
- text-to-video
- stable-diffusion
---

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/637a6daf7ce76c3b83497ea2/ux_sZKB9snVPsKRT1TzfG.gif)

<font size="32">**Try Hotshot-XL yourself here**: https://www.hotshot.co</font>

Hotshot-XL is an AI text-to-GIF model trained to work alongside [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

Hotshot-XL can generate GIFs with any fine-tuned SDXL model. This means two things:
  1. You’ll be able to make GIFs with any existing or newly fine-tuned SDXL model you may want to use.
  2. If you'd like to make GIFs of personalized subjects, you can load your own SDXL based LORAs, and not have to worry about fine-tuning Hotshot-XL. This is awesome because it’s usually much easier to find suitable images for training data than it is to find videos. It also hopefully fits into everyone's existing LORA usage/workflows :) See more [here](https://github.com/hotshotco/Hotshot-XL/blob/main/README.md#text-to-gif-with-personalized-loras).

Hotshot-XL is compatible with SDXL ControlNet to make GIFs in the composition/layout you’d like. See [here](https://github.com/hotshotco/Hotshot-XL/blob/main/README.md#text-to-gif-with-controlnet) for more info.

Hotshot-XL was trained to generate 1 second GIFs at 8 FPS.

Hotshot-XL was trained on various aspect ratios. For best results with the base Hotshot-XL model, we recommend using it with an SDXL model that has been fine-tuned with 512x512 images. You can find an SDXL model we fine-tuned for 512x512 resolutions [here](https://github.com/hotshotco/Hotshot-XL/blob/main/README.md#text-to-gif-with-personalized-loras).



![image/gif](https://cdn-uploads.huggingface.co/production/uploads/637a6daf7ce76c3b83497ea2/XXgnk14nIasPdkvkPlDzn.gif)
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/637a6daf7ce76c3b83497ea2/6OknWOlsl9Zs_esGtPTlZ.jpeg)

Source code is available at https://github.com/hotshotco/Hotshot-XL.

# Model Description
- **Developed by**: Natural Synthetics Inc.
- **Model type**: Diffusion-based text-to-GIF generative model
- **License**: [CreativeML Open RAIL++-M License](https://huggingface.co/hotshotco/Hotshot-XL/raw/main/LICENSE.md)
- **Model Description**: This is a model that can be used to generate and modify GIFs based on text prompts. It is a Latent Diffusion Model that uses two fixed, pretrained text encoders (OpenCLIP-ViT/G and CLIP-ViT/L).
- **Resources for more information**: Check out our [GitHub Repository](https://github.com/hotshotco/Hotshot-XL).


# Limitations and Bias
## Limitations
- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.

## Bias
While the capabilities of video generation models are impressive, they can also reinforce or exacerbate social biases.