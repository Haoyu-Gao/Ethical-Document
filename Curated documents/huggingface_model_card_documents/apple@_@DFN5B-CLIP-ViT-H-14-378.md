---
license: other
license_name: apple-sample-code-license
license_link: LICENSE
---
A CLIP (Contrastive Language-Image Pre-training) model trained on DFN-5B. 
Data Filtering Networks (DFNs) are small networks used to automatically filter large pools of uncurated data. 
This model was trained on 5B images that were filtered from a pool of 43B uncurated image-text pairs 
(12.8B image-text pairs from CommonPool-12.8B + 30B additional public image-text pairs).

This model has been converted to PyTorch from the original JAX checkpoints from Axlearn (https://github.com/apple/axlearn). 
These weights are directly usable in OpenCLIP (image + text).


## Model Details

- **Model Type:**  Contrastive Image-Text, Zero-Shot Image Classification.
- **Dataset:** DFN-5b
- **Papers:**
  - Data Filtering Networks: https://arxiv.org/abs/2309.17425
- **Samples Seen:** 39B (224 x 224) + 5B (384 x 384)
## Model Metrics 
| dataset                |   metric |
|:-----------------------|---------:|
| ImageNet 1k            | 0.84218  |
| Caltech-101            | 0.954479 |
| CIFAR-10               | 0.9879   |
| CIFAR-100              | 0.9041   |
| CLEVR Counts           | 0.362467 |
| CLEVR Distance         | 0.206067 |
| Country211             | 0.37673  |
| Describable Textures   | 0.71383  |
| EuroSAT                | 0.608333 |
| FGVC Aircraft          | 0.719938 |
| Food-101               | 0.963129 |
| GTSRB                  | 0.679018 |
| ImageNet Sketch        | 0.73338  |
| ImageNet v2            | 0.7837   |
| ImageNet-A             | 0.7992   |
| ImageNet-O             | 0.3785   |
| ImageNet-R             | 0.937633 |
| KITTI Vehicle Distance | 0.38256  |
| MNIST                  | 0.8372   |
| ObjectNet <sup>1</sup>              | 0.796867 |
| Oxford Flowers-102     | 0.896834 |
| Oxford-IIIT Pet        | 0.966841 |
| Pascal VOC 2007        | 0.826255 |
| PatchCamelyon          | 0.695953 |
| Rendered SST2          | 0.566722 |
| RESISC45               | 0.755079 |
| Stanford Cars          | 0.959955 |
| STL-10                 | 0.991125 |
| SUN397                 | 0.772799 |
| SVHN                   | 0.671251 |
| Flickr                 | 0.8808   |
| MSCOCO                 | 0.636889 |
| WinoGAViL              | 0.571813 |
| iWildCam               | 0.224911 |
| Camelyon17             | 0.711536 |
| FMoW                   | 0.209024 |
| Dollar Street          | 0.71729  |
| GeoDE                  | 0.935699 |
| **Average**                | **0.709421** |


[1]: Center-crop pre-processing used for ObjectNet (squashing results in lower accuracy of 0.737)
## Model Usage
### With OpenCLIP
```
import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer 

model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
tokenizer = get_tokenizer('ViT-H-14')

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)

labels_list = ["a dog", "a cat", "a donut", "a beignet"]
text = tokenizer(labels_list, context_length=model.context_length)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)

zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)
```

## Citation
```bibtex
@article{fang2023data,
  title={Data Filtering Networks},
  author={Fang, Alex and Jose, Albin Madappally and Jain, Amit and Schmidt, Ludwig and Toshev, Alexander and Shankar, Vaishaal},
  journal={arXiv preprint arXiv:2309.17425},
  year={2023}
}

```