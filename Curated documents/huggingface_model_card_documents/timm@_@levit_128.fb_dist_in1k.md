---
license: apache-2.0
tags:
- image-classification
- timm
datasets:
- imagenet-1k
library_tag: timm
---
# Model card for levit_128.fb_dist_in1k

A LeViT image classification model using convolutional mode (using nn.Conv2d and nn.BatchNorm2d). Pretrained on ImageNet-1k using distillation by paper authors.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 9.2
  - GMACs: 0.4
  - Activations (M): 2.7
  - Image size: 224 x 224
- **Papers:**
  - LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference: https://arxiv.org/abs/2104.01136
- **Original:** https://github.com/facebookresearch/LeViT
- **Dataset:** ImageNet-1k

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(
    urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'))

model = timm.create_model('levit_128.fb_dist_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(
    urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'))

model = timm.create_model(
    'levit_128.fb_dist_in1k',
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
# output is unpooled (ie.e a (batch_size, num_features, H, W) tensor

output = model.forward_head(output, pre_logits=True)
# output is (batch_size, num_features) tensor
```

## Model Comparison
|model                              |top1  |top5  |param_count|img_size|
|-----------------------------------|------|------|-----------|--------|
|levit_384.fb_dist_in1k             |82.596|96.012|39.13      |224     |
|levit_conv_384.fb_dist_in1k        |82.596|96.012|39.13      |224     |
|levit_256.fb_dist_in1k             |81.512|95.48 |18.89      |224     |
|levit_conv_256.fb_dist_in1k        |81.512|95.48 |18.89      |224     |
|levit_conv_192.fb_dist_in1k        |79.86 |94.792|10.95      |224     |
|levit_192.fb_dist_in1k             |79.858|94.792|10.95      |224     |
|levit_128.fb_dist_in1k             |78.474|94.014|9.21       |224     |
|levit_conv_128.fb_dist_in1k        |78.474|94.02 |9.21       |224     |
|levit_128s.fb_dist_in1k            |76.534|92.864|7.78       |224     |
|levit_conv_128s.fb_dist_in1k       |76.532|92.864|7.78       |224     |

## Citation
```bibtex
@InProceedings{Graham_2021_ICCV,
  author    = {Graham, Benjamin and El-Nouby, Alaaeldin and Touvron, Hugo and Stock, Pierre and Joulin, Armand and Jegou, Herve and Douze, Matthijs},
  title     = {LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {12259-12269}
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
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
