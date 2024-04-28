---
license: apache-2.0
tags:
- image-classification
- timm
datasets:
- imagenet-1k
library_tag: timm
---
# Model card for nfnet_l0.ra2_in1k

A NFNet-Lite (Lightweight NFNet) image classification model. Trained in `timm` by Ross Wightman.

Normalization Free Networks are (pre-activation) ResNet-like models without any normalization layers. Instead of Batch Normalization or alternatives, they use Scaled Weight Standardization and specifically placed scalar gains in residual path and at non-linearities based on signal propagation analysis.

Lightweight NFNets are `timm` specific variants that reduce the SE and bottleneck ratio from 0.5 -> 0.25 (reducing widths) and use a smaller group size while maintaining the same depth. SiLU activations used instead of GELU.



## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 35.1
  - GMACs: 4.4
  - Activations (M): 10.5
  - Image size: train = 224 x 224, test = 288 x 288
- **Papers:**
  - High-Performance Large-Scale Image Recognition Without Normalization: https://arxiv.org/abs/2102.06171
  - Characterizing signal propagation to close the performance gap in unnormalized ResNets: https://arxiv.org/abs/2101.08692
- **Original:** https://github.com/huggingface/pytorch-image-models
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

model = timm.create_model('nfnet_l0.ra2_in1k', pretrained=True)
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
    'nfnet_l0.ra2_in1k',
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
    #  torch.Size([1, 1536, 14, 14])
    #  torch.Size([1, 2304, 7, 7])

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
    'nfnet_l0.ra2_in1k',
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
# output is unpooled, a (1, 2304, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).


## Citation
```bibtex
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:2102.06171},
  year={2021}
}
```
```bibtex
@inproceedings{brock2021characterizing,
  author={Andrew Brock and Soham De and Samuel L. Smith},
  title={Characterizing signal propagation to close the performance gap in
  unnormalized ResNets},
  booktitle={9th International Conference on Learning Representations, {ICLR}},
  year={2021}
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
