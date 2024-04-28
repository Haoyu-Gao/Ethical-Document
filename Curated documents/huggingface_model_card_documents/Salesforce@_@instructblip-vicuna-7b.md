---
language: en
license: other
tags:
- vision
- image-captioning
pipeline_tag: image-to-text
---

# InstructBLIP model

InstructBLIP model using [Vicuna-7b](https://github.com/lm-sys/FastChat#model-weights) as language model. InstructBLIP was introduced in the paper [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) by Dai et al.

Disclaimer: The team releasing InstructBLIP did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

InstructBLIP is a visual instruction tuned version of [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2). Refer to the paper for details.

![InstructBLIP architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg)

## Intended uses & limitations

Usage is as follows:

```
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompt = "What is unusual about this image?"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
```

### How to use

For code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/instructblip).