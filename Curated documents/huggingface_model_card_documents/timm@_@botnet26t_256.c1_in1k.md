---
license: apache-2.0
library_name: timm
tags:
- image-classification
- timm
datasets:
- imagenet-1k
---
# Model card for botnet26t_256.c1_in1k

A BotNet image classification model (, based on ResNet architecture). Trained on ImageNet-1k in `timm` by Ross Wightman.

NOTE: this model did not adhere to any specific paper configuration, it was tuned for reasonable training times and reduced frequency of self-attention blocks.

Recipe details:
 * Based on [ResNet Strikes Back](https://arxiv.org/abs/2110.00476) `C` recipes
 * SGD (w/ Nesterov) optimizer and AGC (adaptive gradient clipping).
 * Cosine LR schedule with warmup

This model architecture is implemented using `timm`'s flexible [BYOBNet (Bring-Your-Own-Blocks Network)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/byobnet.py).

BYOB (with BYOANet attention specific blocks) allows configuration of:
 * block / stage layout
 * block-type interleaving
 * stem layout
 * output stride (dilation)
 * activation and norm layers
 * channel and spatial / self-attention layers

...and also includes `timm` features common to many other architectures, including:
 * stochastic depth
 * gradient checkpointing
 * layer-wise LR decay
 * per-stage feature extraction


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 12.5
  - GMACs: 3.3
  - Activations (M): 12.0
  - Image size: 256 x 256
- **Papers:**
  - Bottleneck Transformers for Visual Recognition: https://arxiv.org/abs/2101.11605
  - ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
- **Dataset:** ImageNet-1k

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('botnet26t_256.c1_in1k', pretrained=True)
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
    'botnet26t_256.c1_in1k',
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
    #  torch.Size([1, 64, 128, 128])
    #  torch.Size([1, 256, 64, 64])
    #  torch.Size([1, 512, 32, 32])
    #  torch.Size([1, 1024, 16, 16])
    #  torch.Size([1, 2048, 8, 8])

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
    'botnet26t_256.c1_in1k',
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
# output is unpooled, a (1, 2048, 8, 8) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
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
```bibtex
@article{Srinivas2021BottleneckTF,
  title={Bottleneck Transformers for Visual Recognition},
  author={A. Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and P. Abbeel and Ashish Vaswani},
  journal={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
  pages={16514-16524}
}
```
```bibtex
@inproceedings{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and Jegou, Herve},
  booktitle={NeurIPS 2021 Workshop on ImageNet: Past, Present, and Future}
}
```
