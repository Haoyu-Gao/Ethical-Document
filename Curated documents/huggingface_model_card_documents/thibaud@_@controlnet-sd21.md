---
language:
- en
license: other
tags:
- art
- diffusers
- stable diffusion
- controlnet
datasets: laion/laion-art
---
Want to support my work: you can bought my Artbook: https://thibaud.art 
___

Here's the first version of controlnet for stablediffusion 2.1
Trained on a subset of laion/laion-art

License: refers to the different preprocessor's ones.


### Safetensors version uploaded, only 700mb!

### Canny:
![<canny> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_canny.png)

### Depth:
![<depth> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_depth.png)

### ZoeDepth:
![<depth> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_zoedepth.png)

### Hed:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_hed.png)

### Scribble:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_scribble.png)

### OpenPose:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_openpose.png)

### Color:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_color.png)

### OpenPose:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_openposev2.png)

### LineArt:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_lineart.png)

### Ade20K:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_ade20k.png)

### Normal BAE:
![<hed> 0](https://huggingface.co/thibaud/controlnet-sd21/resolve/main/example_normalbae.png)

### To use with Automatic1111:
* Download the ckpt files or safetensors ones
* Put it in extensions/sd-webui-controlnet/models
* in settings/controlnet, change cldm_v15.yaml by cldm_v21.yaml
* Enjoy

### To use ZoeDepth:
You can use it with annotator depth/le_res but it works better with ZoeDepth Annotator. My PR is not accepted yet but you can use my fork.
My fork: https://github.com/thibaudart/sd-webui-controlnet 
The PR: https://github.com/Mikubill/sd-webui-controlnet/pull/655#issuecomment-1481724024

### Misuse, Malicious Use, and Out-of-Scope Use

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.


Thanks https://huggingface.co/lllyasviel/ for the implementation and the release of 1.5 models.
Thanks https://huggingface.co/p1atdev/ for the conversion script from ckpt to safetensors pruned & fp16


### Models can't be sell, merge, distributed without prior writing agreement.

