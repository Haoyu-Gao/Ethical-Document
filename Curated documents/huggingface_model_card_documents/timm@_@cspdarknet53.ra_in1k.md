---
license: apache-2.0
tags:
- image-classification
- timm
library_tag: timm
---
# Model card for cspdarknet53.ra_in1k

A CSP-DarkNet (Cross-Stage-Partial) image classification model. Trained on ImageNet-1k in `timm` using recipe template described below.

Recipe details:
 * RandAugment `RA` recipe. Inspired by and evolved from EfficientNet RandAugment recipes. Published as `B` recipe in [ResNet Strikes Back](https://arxiv.org/abs/2110.00476).
 * RMSProp (TF 1.0 behaviour) optimizer, EMA weight averaging
 * Step (exponential decay w/ staircase) LR schedule with warmup


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 27.6
  - GMACs: 6.6
  - Activations (M): 16.8
  - Image size: 256 x 256
- **Papers:**
  - CSPNet: A New Backbone that can Enhance Learning Capability of CNN: https://arxiv.org/abs/1911.11929
  - YOLOv3: An Incremental Improvement: https://arxiv.org/abs/1804.02767
  - ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
- **Original:** https://github.com/huggingface/pytorch-image-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('cspdarknet53.ra_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'cspdarknet53.ra_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 32, 256, 256])
    #  torch.Size([1, 64, 128, 128])
    #  torch.Size([1, 128, 64, 64])
    #  torch.Size([1, 256, 32, 32])
    #  torch.Size([1, 512, 16, 16])
    #  torch.Size([1, 1024, 8, 8])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'cspdarknet53.ra_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1024, 8, 8) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@article{Wang2019CSPNetAN,
  title={CSPNet: A New Backbone that can Enhance Learning Capability of CNN},
  author={Chien-Yao Wang and Hong-Yuan Mark Liao and I-Hau Yeh and Yueh-Hua Wu and Ping-Yang Chen and Jun-Wei Hsieh},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2019},
  pages={1571-1580}
}
```
```bibtex
@article{Redmon2018YOLOv3AI,
  title={YOLOv3: An Incremental Improvement},
  author={Joseph Redmon and Ali Farhadi},
  journal={ArXiv},
  year={2018},
  volume={abs/1804.02767}
}
```
```bibtex
@inproceedings{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and Jegou, Herve},
  booktitle={NeurIPS 2021 Workshop on ImageNet: Past, Present, and Future}
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
