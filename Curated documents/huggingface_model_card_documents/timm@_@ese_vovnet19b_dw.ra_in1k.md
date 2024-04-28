---
license: apache-2.0
library_name: timm
tags:
- image-classification
- timm
datasets:
- imagenet-1k
---
# Model card for ese_vovnet19b_dw.ra_in1k

A VoVNet-v2 image classification model. Pretrained on ImageNet-1k in `timm` by Ross Wightman using RandAugment `RA` recipe. Related to `B` recipe in [ResNet Strikes Back](https://arxiv.org/abs/2110.00476).

## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 6.5
  - GMACs: 1.3
  - Activations (M): 8.2
  - Image size: train = 224 x 224, test = 288 x 288
- **Papers:**
  - An Energy and GPU-Computation Efficient Backbone Network: https://arxiv.org/abs/1904.09730
  - CenterMask : Real-Time Anchor-Free Instance Segmentation: https://arxiv.org/abs/1911.06667
  - ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
- **Dataset:** ImageNet-1k
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

model = timm.create_model('ese_vovnet19b_dw.ra_in1k', pretrained=True)
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
    'ese_vovnet19b_dw.ra_in1k',
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
    #  torch.Size([1, 64, 112, 112])
    #  torch.Size([1, 256, 56, 56])
    #  torch.Size([1, 512, 28, 28])
    #  torch.Size([1, 768, 14, 14])
    #  torch.Size([1, 1024, 7, 7])

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
    'ese_vovnet19b_dw.ra_in1k',
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
# output is unpooled, a (1, 1024, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Citation
```bibtex
@inproceedings{lee2019energy,
  title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
  author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2019}
}
```
```bibtex
@article{lee2019centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  booktitle={CVPR},
  year={2020}
}
```
```bibtex
@inproceedings{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and Jegou, Herve},
  booktitle={NeurIPS 2021 Workshop on ImageNet: Past, Present, and Future}
}
```
