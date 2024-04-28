---
license: mit
tags:
- image-classification
- timm
datasets:
- imagenet-1k
library_tag: timm
---
# Model card for rexnet_100.nav_in1k

A ReXNet image classification model. Pretrained on ImageNet-1k by paper authors.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 4.8
  - GMACs: 0.4
  - Activations (M): 7.4
  - Image size: 224 x 224
- **Papers:**
  - Rethinking Channel Dimensions for Efficient Model Design: https://arxiv.org/abs/2007.00992
- **Original:** https://github.com/clovaai/rexnet
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

model = timm.create_model('rexnet_100.nav_in1k', pretrained=True)
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
    'rexnet_100.nav_in1k',
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
    #  torch.Size([1, 38, 56, 56])
    #  torch.Size([1, 61, 28, 28])
    #  torch.Size([1, 128, 14, 14])
    #  torch.Size([1, 185, 7, 7])

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
    'rexnet_100.nav_in1k',
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
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results)."

|model                    |top1  |top5  |param_count|img_size|crop_pct|
|-------------------------|------|------|-----------|--------|--------|
|rexnetr_300.sw_in12k_ft_in1k|84.53 |97.252|34.81      |288     |1.0     |
|rexnetr_200.sw_in12k_ft_in1k|83.164|96.648|16.52      |288     |1.0     |
|rexnet_300.nav_in1k      |82.772|96.232|34.71      |224     |0.875   |
|rexnet_200.nav_in1k      |81.652|95.668|16.37      |224     |0.875   |
|rexnet_150.nav_in1k      |80.308|95.174|9.73       |224     |0.875   |
|rexnet_130.nav_in1k      |79.478|94.68 |7.56       |224     |0.875   |
|rexnet_100.nav_in1k      |77.832|93.886|4.8        |224     |0.875   |

## Citation
```bibtex
@misc{han2021rethinking,
  title={Rethinking Channel Dimensions for Efficient Model Design}, 
  author={Dongyoon Han and Sangdoo Yun and Byeongho Heo and YoungJoon Yoo},
  year={2021},
  eprint={2007.00992},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
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
