---
license: apache-2.0
library_name: diffusers
tags:
- text-to-image
- ultra-realistic
- text-to-image
- stable-diffusion
- distilled-model
- knowledge-distillation
datasets:
- zzliang/GRIT
- wanng/midjourney-v5-202304-clean
pinned: true
---

# Segmind-Vega Model Card


## Demo

Try out the Segmind-Vega model at [Segmind-Vega](https://www.segmind.com/models/segmind-vega) for ⚡ fastest inference.

## Model Description

The Segmind-Vega Model is a distilled version of the Stable Diffusion XL (SDXL), offering a remarkable **70% reduction in size** and an impressive **100% speedup** while retaining high-quality text-to-image generation capabilities. Trained on diverse datasets, including Grit and Midjourney scrape data, it excels at creating a wide range of visual content based on textual prompts.

Employing a knowledge distillation strategy, Segmind-Vega leverages the teachings of several expert models, including SDXL, ZavyChromaXL, and JuggernautXL, to combine their strengths and produce compelling visual outputs.

## Image Comparison (Segmind-Vega vs SDXL)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/7vsFKKg5xAqvEEBtZf85q.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/gDFFMfaCUnntO8JfxhC__.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/bZylkXH3PhFhLYJWG6WJ5.png)

## Speed Comparison (Segmind-Vega vs SD-1.5 vs SDXL)

The tests were conducted on an A100 80GB GPU.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/CGfID3b640dXnlOQL_k28.png)
(Note: All times are reported with the respective tiny-VAE!)
## Parameters Comparison (Segmind-Vega vs SD-1.5 vs SDXL)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/tyu_7S9uQlC3r_3OrvNmU.png)

## Usage:
This model can be used via the 🧨 Diffusers library. 

Make sure to install diffusers by running
```bash
pip install diffusers
```

In addition, please install `transformers`, `safetensors`, and `accelerate`:
```bash
pip install transformers accelerate safetensors
```

To use the model, you can run the following:

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("segmind/Segmind-Vega", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
prompt = "A cute cat eating a slice of pizza, stunning color scheme, masterpiece, illustration" # Your prompt here
neg_prompt = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)" # Negative prompt here
image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
```

### Please do use negative prompting and a CFG around 9.0 for the best quality!

### Model Description

- **Developed by:** [Segmind](https://www.segmind.com/)
- **Developers:** [Yatharth Gupta](https://huggingface.co/Warlord-K) and [Vishnu Jaddipal](https://huggingface.co/Icar).
- **Model type:** Diffusion-based text-to-image generative model
- **License:** Apache 2.0
- **Distilled From:** [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### Key Features

- **Text-to-Image Generation:** The Segmind-Vega model excels at generating images from text prompts, enabling a wide range of creative applications.

- **Distilled for Speed:** Designed for efficiency, this model offers an impressive 100% speedup, making it suitable for real-time applications and scenarios where rapid image generation is essential.

- **Diverse Training Data:** Trained on diverse datasets, the model can handle a variety of textual prompts and generate corresponding images effectively.

- **Knowledge Distillation:** By distilling knowledge from multiple expert models, the Segmind-Vega Model combines their strengths and minimizes their limitations, resulting in improved performance.

### Model Architecture

The Segmind-Vega Model is a compact version with a remarkable 70% reduction in size compared to the Base SDXL Model.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/SU3x12sNRrpxF7wjXvLMl.png)

### Training Info

These are the key hyperparameters used during training:

- Steps: 540,000
- Learning rate: 1e-5
- Batch size: 16
- Gradient accumulation steps: 8
- Image resolution: 1024
- Mixed-precision: fp16


### Model Sources

For research and development purposes, the Segmind-Vega Model can be accessed via the Segmind AI platform. For more information and access details, please visit [Segmind](https://www.segmind.com/models/Segmind-Vega).

## Uses

### Direct Use

The Segmind-Vega Model is suitable for research and practical applications in various domains, including:

- **Art and Design:** It can be used to generate artworks, designs, and other creative content, providing inspiration and enhancing the creative process.

- **Education:** The model can be applied in educational tools to create visual content for teaching and learning purposes.

- **Research:** Researchers can use the model to explore generative models, evaluate its performance, and push the boundaries of text-to-image generation.

- **Safe Content Generation:** It offers a safe and controlled way to generate content, reducing the risk of harmful or inappropriate outputs.

- **Bias and Limitation Analysis:** Researchers and developers can use the model to probe its limitations and biases, contributing to a better understanding of generative models' behavior.

### Downstream Use

The Segmind-Vega Model can also be used directly with the 🧨 Diffusers library training scripts for further training, including:

- **[LoRA](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py):**
```bash
export MODEL_NAME="segmind/Segmind-Vega"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="vega-pokemon-model-lora" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  --push_to_hub
```

- **[Fine-Tune](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py):**
```bash
export MODEL_NAME="segmind/Segmind-Vega"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=1024 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="vega-pokemon-model" \
  --push_to_hub
```

- **[Dreambooth LoRA](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py):**
```bash
export MODEL_NAME="segmind/Segmind-Vega"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-vega"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

### Out-of-Scope Use

The Segmind-Vega Model is not suitable for creating factual or accurate representations of people, events, or real-world information. It is not intended for tasks requiring high precision and accuracy.

## Limitations and Bias

**Limitations & Bias:**
The Segmind-Vega Model faces challenges in achieving absolute photorealism, especially in human depictions. While it may encounter difficulties in incorporating clear text and maintaining the fidelity of complex compositions due to its autoencoding approach, these challenges present opportunities for future enhancements. Importantly, the model's exposure to a diverse dataset, though not a cure-all for ingrained societal and digital biases, represents a foundational step toward more equitable technology. Users are encouraged to interact with this pioneering tool with an understanding of its current limitations, fostering an environment of conscious engagement and anticipation for its continued evolution.
