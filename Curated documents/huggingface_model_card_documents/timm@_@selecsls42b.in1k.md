---
license: cc-by-4.0
library_name: timm
tags:
- image-classification
- timm
datasets:
- imagenet-1k
---
# Model card for selecsls42b.in1k

A SelecSLS image classification model. Trained on ImageNet-1k by paper authors.

## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 32.5
  - GMACs: 3.0
  - Activations (M): 4.6
  - Image size: 224 x 224
- **Papers:**
  - XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera: https://arxiv.org/abs/1907.00837
- **Dataset:** ImageNet-1k
- **Original:** https://github.com/mehtadushy/SelecSLS-Pytorch

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('selecsls42b.in1k', pretrained=True)
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
    'selecsls42b.in1k',
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
    #  torch.Size([1, 32, 112, 112])
    #  torch.Size([1, 128, 56, 56])
    #  torch.Size([1, 288, 28, 28])
    #  torch.Size([1, 480, 14, 14])
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
    'selecsls42b.in1k',
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
# output is unpooled, a (1, 1024, 4, 4) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@inproceedings{XNect_SIGGRAPH2020,
  author = {Mehta, Dushyant and Sotnychenko, Oleksandr and Mueller, Franziska and Xu, Weipeng and Elgharib, Mohamed and Fua, Pascal and Seidel, Hans-Peter and Rhodin, Helge and Pons-Moll, Gerard and Theobalt, Christian},
  title = {{XNect}: Real-time Multi-Person {3D} Motion Capture with a Single {RGB} Camera},
  journal = {ACM Transactions on Graphics},
  url = {http://gvv.mpi-inf.mpg.de/projects/XNect/},
  numpages = {17},
  volume={39},
  number={4},
  month = July,
  year = {2020},
  doi={10.1145/3386569.3392410}
} 
```
