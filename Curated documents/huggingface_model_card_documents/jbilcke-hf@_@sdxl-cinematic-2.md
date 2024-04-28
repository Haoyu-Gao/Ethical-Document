---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
datasets:
- jbilcke-hf/cinematic-2
base_model: stabilityai/stable-diffusion-xl-base-1.0
instance_prompt: cinematic-2
inference: true
---
    
# LoRA DreamBooth - jbilcke-hf/sdxl-cinematic-2
These are LoRA adaption weights for stabilityai/stable-diffusion-xl-base-1.0 trained on @fffiloni's SD-XL trainer. 
The weights were trained on the concept prompt: 
```
cinematic-2
```  
Use this keyword to trigger your custom model in your prompts. 
LoRA for the text encoder was enabled: False.
Special VAE used for training: madebyollin/sdxl-vae-fp16-fix.
## Usage
Make sure to upgrade diffusers to >= 0.19.0:
```
pip install diffusers --upgrade
```
In addition make sure to install transformers, safetensors, accelerate as well as the invisible watermark:
```
pip install invisible_watermark transformers accelerate safetensors
```
To just use the base model, you can run:
```python
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True
)
pipe.to(device)
# This is where you load your trained weights
specific_safetensors = "pytorch_lora_weights.safetensors"
lora_scale = 0.9
pipe.load_lora_weights(
    'jbilcke-hf/sdxl-cinematic-2', 
    weight_name = specific_safetensors,
    # use_auth_token = True 
)
prompt = "A majestic cinematic-2 jumping from a big stone at night"
image = pipe(
    prompt=prompt, 
    num_inference_steps=50,
    cross_attention_kwargs={"scale": lora_scale}
).images[0]
```
