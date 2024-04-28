---
license: other
library_name: timm
tags:
- image-classification
- timm
datasets:
- imagenet-1k
---
# Model card for mobilevitv2_200.cvnets_in22k_ft_in1k_384

A MobileViT-v2 image classification model. Pretrained on ImageNet-22k and fine-tuned on ImageNet-1k by paper authors.

See license details at https://github.com/apple/ml-cvnets/blob/main/LICENSE

## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 18.4
  - GMACs: 16.2
  - Activations (M): 72.3
  - Image size: 384 x 384
- **Papers:**
  - Separable Self-attention for Mobile Vision Transformers: https://arxiv.org/abs/2206.02680
- **Original:** https://github.com/apple/ml-cvnets
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

model = timm.create_model('mobilevitv2_200.cvnets_in22k_ft_in1k_384', pretrained=True)
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
    'mobilevitv2_200.cvnets_in22k_ft_in1k_384',
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
    #  torch.Size([1, 128, 192, 192])
    #  torch.Size([1, 256, 96, 96])
    #  torch.Size([1, 512, 48, 48])
    #  torch.Size([1, 768, 24, 24])
    #  torch.Size([1, 1024, 12, 12])

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
    'mobilevitv2_200.cvnets_in22k_ft_in1k_384',
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
# output is unpooled, a (1, 1024, 12, 12) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@article{Mehta2022SeparableSF,
  title={Separable Self-attention for Mobile Vision Transformers},
  author={Sachin Mehta and Mohammad Rastegari},
  journal={ArXiv},
  year={2022},
  volume={abs/2206.02680}
}
```
