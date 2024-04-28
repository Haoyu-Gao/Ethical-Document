---
license: other
pipeline_tag: image-to-video
license_name: stable-video-diffusion-nc-community
license_link: LICENSE
---

# Stable Video Diffusion Image-to-Video Model Card

<!-- Provide a quick summary of what the model is/does. -->
![row01](output_tile.gif)
Stable Video Diffusion (SVD) Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a video from it. 

## Model Details

### Model Description

(SVD) Image-to-Video is a latent diffusion model trained to generate short video clips from an image conditioning. 
This model was trained to generate 14 frames at resolution 576x1024 given a context frame of the same size.
We also finetune the widely used [f8-decoder](https://huggingface.co/docs/diffusers/api/models/autoencoderkl#loading-from-the-original-format) for temporal consistency. 
For convenience, we additionally provide the model with the 
standard frame-wise decoder [here](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/svd_image_decoder.safetensors).


- **Developed by:** Stability AI
- **Funded by:** Stability AI
- **Model type:** Generative image-to-video model

### Model Sources

For research purposes, we recommend our `generative-models` Github repository (https://github.com/Stability-AI/generative-models), 
which implements the most popular diffusion frameworks (both training and inference).

- **Repository:** https://github.com/Stability-AI/generative-models
- **Paper:** https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets

## Evaluation
![comparison](comparison.png)
The chart above evaluates user preference for SVD-Image-to-Video over [GEN-2](https://research.runwayml.com/gen2) and [PikaLabs](https://www.pika.art/).
SVD-Image-to-Video is preferred by human voters in terms of video quality. For details on the user study, we refer to the [research paper](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets)

## Uses

### Direct Use

The model is intended for research purposes only. Possible research areas and tasks include

- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, 
and therefore using the model to generate such content is out-of-scope for the abilities of this model.
The model should not be used in any way that violates Stability AI's [Acceptable Use Policy](https://stability.ai/use-policy).

## Limitations and Bias

### Limitations
- The generated videos are rather short (<= 4sec), and the model does not achieve perfect photorealism.
- The model may generate videos without motion, or very slow camera pans.
- The model cannot be controlled through text.
- The model cannot render legible text.
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.


### Recommendations

The model is intended for research purposes only.

## How to Get Started with the Model

Check out https://github.com/Stability-AI/generative-models



