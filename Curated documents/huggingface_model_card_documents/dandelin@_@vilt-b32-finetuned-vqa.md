---
license: apache-2.0
tags:
- visual-question-answering
widget:
- text: What's the animal doing?
  src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
- text: What is on top of the building?
  src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
---

# Vision-and-Language Transformer (ViLT), fine-tuned on VQAv2

Vision-and-Language Transformer (ViLT) model fine-tuned on [VQAv2](https://visualqa.org/). It was introduced in the paper [ViLT: Vision-and-Language Transformer
Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) by Kim et al. and first released in [this repository](https://github.com/dandelin/ViLT). 

Disclaimer: The team releasing ViLT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Intended uses & limitations

You can use the raw model for visual question answering. 

### How to use

Here is how to use this model in PyTorch:

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```

## Training data

(to do)

## Training procedure

### Preprocessing

(to do)

### Pretraining

(to do)

## Evaluation results

(to do)

### BibTeX entry and citation info

```bibtex
@misc{kim2021vilt,
      title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}, 
      author={Wonjae Kim and Bokyung Son and Ildoo Kim},
      year={2021},
      eprint={2102.03334},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```