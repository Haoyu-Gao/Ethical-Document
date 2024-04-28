---
license: apache-2.0
tags:
- object-detection
- vision
datasets:
- coco
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg
  example_title: Savanna
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg
  example_title: Football Match
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg
  example_title: Airport
---

# YOLOS (small-sized) model

YOLOS model fine-tuned on COCO 2017 object detection (118k annotated images). It was introduced in the paper [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) by Fang et al. and first released in [this repository](https://github.com/hustvl/YOLOS). 

Disclaimer: The team releasing YOLOS did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

YOLOS is a Vision Transformer (ViT) trained using the DETR loss. Despite its simplicity, a base-sized YOLOS model is able to achieve 42 AP on COCO validation 2017 (similar to DETR and more complex frameworks such as Faster R-CNN).

The model is trained using a "bipartite matching loss": one compares the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a "no object" as class and "no bounding box" as bounding box). The Hungarian matching algorithm is used to create an optimal one-to-one mapping between each of the N queries and each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.

## Intended uses & limitations

You can use the raw model for object detection. See the [model hub](https://huggingface.co/models?search=hustvl/yolos) to look for all available YOLOS models.

### How to use

Here is how to use this model:

```python
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes
```

Currently, both the feature extractor and model support PyTorch. 

## Training data

The YOLOS model was pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet2012) and fine-tuned on [COCO 2017 object detection](https://cocodataset.org/#download), a dataset consisting of 118k/5k annotated images for training/validation respectively. 

### Training

The model was pre-trained for 200 epochs on ImageNet-1k and fine-tuned for 150 epochs on COCO.

## Evaluation results

This model achieves an AP (average precision) of **36.1** on COCO 2017 validation. For more details regarding evaluation results, we refer to table 1 of the original paper.

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2106-00666,
  author    = {Yuxin Fang and
               Bencheng Liao and
               Xinggang Wang and
               Jiemin Fang and
               Jiyang Qi and
               Rui Wu and
               Jianwei Niu and
               Wenyu Liu},
  title     = {You Only Look at One Sequence: Rethinking Transformer in Vision through
               Object Detection},
  journal   = {CoRR},
  volume    = {abs/2106.00666},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.00666},
  eprinttype = {arXiv},
  eprint    = {2106.00666},
  timestamp = {Fri, 29 Apr 2022 19:49:16 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-00666.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```