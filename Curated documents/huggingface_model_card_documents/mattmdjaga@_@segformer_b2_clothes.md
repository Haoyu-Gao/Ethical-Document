---
license: mit
tags:
- vision
- image-segmentation
datasets:
- mattmdjaga/human_parsing_dataset
widget:
- src: https://images.unsplash.com/photo-1643310325061-2beef64926a5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8cmFjb29uc3xlbnwwfHwwfHw%3D&w=1000&q=80
  example_title: Person
- src: https://freerangestock.com/sample/139043/young-man-standing-and-leaning-on-car.jpg
  example_title: Person
---
# Segformer B2 fine-tuned for clothes segmentation

SegFormer model fine-tuned on [ATR dataset](https://github.com/lemondan/HumanParsing-Dataset) for clothes segmentation but can also be used for human segmentation.
The dataset on hugging face is called "mattmdjaga/human_parsing_dataset".


**NEW** - 
**[Training code](https://github.com/mattmdjaga/segformer_b2_clothes)**. Right now it only contains the pure code with some comments, but soon I'll add a colab notebook version
 and a blog post with it to make it more friendly. 

```python
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"

image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]
plt.imshow(pred_seg)
```

Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"

### Evaluation

|  Label Index  |    Label Name    | Category Accuracy | Category IoU |
|:-------------:|:----------------:|:-----------------:|:------------:|
|       0       |    Background    |       0.99        |     0.99     |
|       1       |        Hat       |       0.73        |     0.68     |
|       2       |        Hair      |       0.91        |     0.82     |
|       3       |    Sunglasses    |       0.73        |     0.63     |
|       4       |  Upper-clothes   |       0.87        |     0.78     |
|       5       |       Skirt      |       0.76        |     0.65     |
|       6       |       Pants      |       0.90        |     0.84     |
|       7       |       Dress      |       0.74        |     0.55     |
|       8       |       Belt       |       0.35        |     0.30     |
|       9       |    Left-shoe     |       0.74        |     0.58     |
|      10       |   Right-shoe     |       0.75        |     0.60     |
|      11       |       Face       |       0.92        |     0.85     |
|      12       |    Left-leg      |       0.90        |     0.82     |
|      13       |   Right-leg      |       0.90        |     0.81     |
|      14       |    Left-arm      |       0.86        |     0.74     |
|      15       |   Right-arm      |       0.82        |     0.73     |
|      16       |        Bag       |       0.91        |     0.84     |
|      17       |       Scarf      |       0.63        |     0.29     |

Overall Evaluation Metrics:
- Evaluation Loss: 0.15
- Mean Accuracy: 0.80
- Mean IoU: 0.69

### License

The license for this model can be found [here](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2105-15203,
  author    = {Enze Xie and
               Wenhai Wang and
               Zhiding Yu and
               Anima Anandkumar and
               Jose M. Alvarez and
               Ping Luo},
  title     = {SegFormer: Simple and Efficient Design for Semantic Segmentation with
               Transformers},
  journal   = {CoRR},
  volume    = {abs/2105.15203},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.15203},
  eprinttype = {arXiv},
  eprint    = {2105.15203},
  timestamp = {Wed, 02 Jun 2021 11:46:42 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-15203.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```