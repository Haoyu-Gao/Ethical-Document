---
license: apache-2.0
tags:
- dino
- vision
datasets:
- imagenet-1k
---

# Vision Transformer (small-sized model, patch size 16) trained using DINO 

Vision Transformer (ViT) model trained using the DINO method. It was introduced in the paper [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) by Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin and first released in [this repository](https://github.com/facebookresearch/dino). 

Disclaimer: The team releasing DINO did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a self-supervised fashion, namely ImageNet-1k, at a resolution of 224x224 pixels. 

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

Note that this model does not include any fine-tuned heads. 

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=google/vit) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model:

```python
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2104-14294,
  author    = {Mathilde Caron and
               Hugo Touvron and
               Ishan Misra and
               Herv{\'{e}} J{\'{e}}gou and
               Julien Mairal and
               Piotr Bojanowski and
               Armand Joulin},
  title     = {Emerging Properties in Self-Supervised Vision Transformers},
  journal   = {CoRR},
  volume    = {abs/2104.14294},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.14294},
  archivePrefix = {arXiv},
  eprint    = {2104.14294},
  timestamp = {Tue, 04 May 2021 15:12:43 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2104-14294.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```