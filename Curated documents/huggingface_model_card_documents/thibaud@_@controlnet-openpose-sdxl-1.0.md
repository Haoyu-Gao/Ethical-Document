---
license: other
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- controlnet
base_model: stabilityai/stable-diffusion-xl-base-1.0
inference: false
---
    
# SDXL-controlnet: OpenPose (v2)

These are controlnet weights trained on stabilityai/stable-diffusion-xl-base-1.0 with OpenPose (v2) conditioning. You can find some example images in the following. 

prompt: a ballerina, romantic sunset, 4k photo
![images_0)](./screenshot_ballerina.png)


### Comfy Workflow
![images_0)](./out_ballerina.png)


(Image is from ComfyUI, you can drag and drop in Comfy to use it as workflow)

License: refers to the OpenPose's one.

### Using in 🧨 diffusers

First, install all the libraries:

```bash
pip install -q controlnet_aux transformers accelerate
pip install -q git+https://github.com/huggingface/diffusers
```

Now, we're ready to make Darth Vader dance:

```python
from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image


# Compute openpose conditioning image.
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(image)

# Initialize ControlNet pipeline.
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()


# Infer.
prompt = "Darth vader dancing in a desert, high quality"
negative_prompt = "low quality, bad quality"
images = pipe(
    prompt, 
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    num_images_per_prompt=4,
    image=openpose_image.resize((1024, 1024)),
    generator=torch.manual_seed(97),
).images
images[0]
```

Here are some gemerated examples:

![](./darth_vader_grid.png)


### Training

Use of the training script by HF🤗 [here](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sdxl.md). 

#### Training data
This checkpoint was first trained for 15,000 steps on laion 6a resized to a max minimum dimension of 768. 

#### Compute
one 1xA100 machine (Thanks a lot HF🤗 to provide the compute!)

#### Batch size
Data parallel with a single gpu batch size of 2 with gradient accumulation 8.

#### Hyper Parameters
Constant learning rate of 8e-5

#### Mixed precision
fp16