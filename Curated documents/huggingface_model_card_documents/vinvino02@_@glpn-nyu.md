---
license: apache-2.0
tags:
- vision
- depth-estimation
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

# GLPN fine-tuned on NYUv2

Global-Local Path Networks (GLPN) model trained on NYUv2 for monocular depth estimation. It was introduced in the paper [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://arxiv.org/abs/2201.07436) by Kim et al. and first released in [this repository](https://github.com/vinvino02/GLPDepth). 

Disclaimer: The team releasing GLPN did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

GLPN uses SegFormer as backbone and adds a lightweight head on top for depth estimation.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg)

## Intended uses & limitations

You can use the raw model for monocular depth estimation. See the [model hub](https://huggingface.co/models?search=glpn) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model:

```python
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# prepare image for the model
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/glpn).

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2201-07436,
  author    = {Doyeon Kim and
               Woonghyun Ga and
               Pyunghwan Ahn and
               Donggyu Joo and
               Sehwan Chun and
               Junmo Kim},
  title     = {Global-Local Path Networks for Monocular Depth Estimation with Vertical
               CutDepth},
  journal   = {CoRR},
  volume    = {abs/2201.07436},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.07436},
  eprinttype = {arXiv},
  eprint    = {2201.07436},
  timestamp = {Fri, 21 Jan 2022 13:57:15 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-07436.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```