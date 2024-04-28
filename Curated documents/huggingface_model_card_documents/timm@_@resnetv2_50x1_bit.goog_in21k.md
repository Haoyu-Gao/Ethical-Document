---
license: apache-2.0
tags:
- image-classification
- timm
datasets:
- imagenet-21k
library_tag: timm
---
# Model card for resnetv2_50x1_bit.goog_in21k

A ResNet-V2-BiT (Big Transfer w/ pre-activation ResNet) image classification model. Trained on ImageNet-21k by paper authors.

This model uses:
* Group Normalization (GN) in combination with Weight Standardization (WS) instead of Batch Normalization (BN)..


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 68.3
  - GMACs: 4.3
  - Activations (M): 11.1
  - Image size: 224 x 224
- **Papers:**
  - Big Transfer (BiT): General Visual Representation Learning: https://arxiv.org/abs/1912.11370
  - Identity Mappings in Deep Residual Networks: https://arxiv.org/abs/1603.05027
- **Dataset:** ImageNet-21k
- **Original:** https://github.com/google-research/big_transfer

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('resnetv2_50x1_bit.goog_in21k', pretrained=True)
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
    'resnetv2_50x1_bit.goog_in21k',
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
    #  torch.Size([1, 1024, 14, 14])
    #  torch.Size([1, 2048, 7, 7])

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
    'resnetv2_50x1_bit.goog_in21k',
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
# output is unpooled, a (1, 2048, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@inproceedings{Kolesnikov2019BigT,
  title={Big Transfer (BiT): General Visual Representation Learning},
  author={Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Joan Puigcerver and Jessica Yung and Sylvain Gelly and Neil Houlsby},
  booktitle={European Conference on Computer Vision},
  year={2019}
}
```
```bibtex
@article{He2016,
  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title = {Identity Mappings in Deep Residual Networks},
  journal = {arXiv preprint arXiv:1603.05027},
  year = {2016}
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
