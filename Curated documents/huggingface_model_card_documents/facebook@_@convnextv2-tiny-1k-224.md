---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

# ConvNeXt V2 (tiny-sized model) 

ConvNeXt V2 model pretrained using the FCMAE framework and fine-tuned on the ImageNet-1K dataset at resolution 224x224. It was introduced in the paper [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) by Woo et al. and first released in [this repository](https://github.com/facebookresearch/ConvNeXt-V2). 

Disclaimer: The team releasing ConvNeXT V2 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

ConvNeXt V2 is a pure convolutional model (ConvNet) that introduces a fully convolutional masked autoencoder framework (FCMAE) and a new Global Response Normalization (GRN) layer to ConvNeXt. ConvNeXt V2 significantly improves the performance of pure ConvNets on various recognition benchmarks.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png)

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=convnextv2) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

preprocessor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")

inputs = preprocessor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/convnextv2).

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2301-00808,
  author    = {Sanghyun Woo and
               Shoubhik Debnath and
               Ronghang Hu and
               Xinlei Chen and
               Zhuang Liu and
               In So Kweon and
               Saining Xie},
  title     = {ConvNeXt {V2:} Co-designing and Scaling ConvNets with Masked Autoencoders},
  journal   = {CoRR},
  volume    = {abs/2301.00808},
  year      = {2023},
  url       = {https://doi.org/10.48550/arXiv.2301.00808},
  doi       = {10.48550/arXiv.2301.00808},
  eprinttype = {arXiv},
  eprint    = {2301.00808},
  timestamp = {Tue, 10 Jan 2023 15:10:12 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2301-00808.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```