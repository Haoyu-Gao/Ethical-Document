---
license: apache-2.0
library_name: timm
tags:
- image-classification
- timm
datasets:
- imagenet-1k
---
# Model card for spnasnet_100.rmsp_in1k

A SPNasNet image classification model. Trained on ImageNet-1k in `timm` using recipe template described below.

Recipe details:
 * A simple RmsProp based recipe without RandAugment. Using RandomErasing, mixup, dropout, standard random-resize-crop augmentation.
 * RMSProp (TF 1.0 behaviour) optimizer, EMA weight averaging
 * Step (exponential decay w/ staircase) LR schedule with warmup


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 4.4
  - GMACs: 0.3
  - Activations (M): 6.0
  - Image size: 224 x 224
- **Papers:**
  - Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours: https://arxiv.org/abs/1904.02877
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

model = timm.create_model('spnasnet_100.rmsp_in1k', pretrained=True)
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
    'spnasnet_100.rmsp_in1k',
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
    #  torch.Size([1, 16, 112, 112])
    #  torch.Size([1, 24, 56, 56])
    #  torch.Size([1, 40, 28, 28])
    #  torch.Size([1, 96, 14, 14])
    #  torch.Size([1, 320, 7, 7])

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
    'spnasnet_100.rmsp_in1k',
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
# output is unpooled, a (1, 1280, 7, 7) shaped tensor

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
@inproceedings{stamoulis2020single,
  title={Single-path nas: Designing hardware-efficient convnets in less than 4 hours},
  author={Stamoulis, Dimitrios and Ding, Ruizhou and Wang, Di and Lymberopoulos, Dimitrios and Priyantha, Bodhi and Liu, Jie and Marculescu, Diana},
  booktitle={Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019, W{"u}rzburg, Germany, September 16--20, 2019, Proceedings, Part II},
  pages={481--497},
  year={2020},
  organization={Springer}
}
```
