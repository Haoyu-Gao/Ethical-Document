---
language: en
license: mit
tags:
- bridgetower
- gaudi
datasets:
- conceptual_captions
- conceptual_12m
- sbu_captions
- visual_genome
- mscoco_captions
---

# BridgeTower large-itm-mlm-itc model

The BridgeTower model was proposed in "BridgeTower: Building Bridges Between Encoders in Vision-Language Representative Learning" by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan. 
The model was pretrained on English language using masked language modeling (MLM) and image text matching (ITM)objectives. It was introduced in
[this paper](https://arxiv.org/pdf/2206.08657.pdf) and first released in
[this repository](https://github.com/microsoft/BridgeTower). 

BridgeTower got accepted to [AAAI'23](https://aaai.org/Conferences/AAAI-23/).

## Model description

The abstract from the paper is the following:
Vision-Language (VL) models with the Two-Tower architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a deep cross-modal encoder, or feed the last-layer uni-modal representations from the deep pre-trained uni-modal encoders into the top cross-modal encoder. Both approaches potentially restrict vision-language representation learning and limit model performance. In this paper, we propose BridgeTower, which introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BridgeTower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets.

## Intended uses & limitations


### How to use

Here is how to use this model to perform contrastive learning between image and text pairs:

```python
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
import requests
from PIL import Image
import torch

image_urls = [
    "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg"]
texts = [
    "two dogs in a car",
    "two cats sleeping on a couch"]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

inputs  = processor(images, texts, padding=True, return_tensors="pt")
outputs = model(**inputs)

inputs  = processor(images, texts[::-1], padding=True, return_tensors="pt")
outputs_swapped = model(**inputs)

print('Loss', outputs.loss.item())
# Loss 0.00191505195107311
print('Loss with swapped images', outputs_swapped.loss.item())
# Loss with swapped images 2.1259872913360596 
```

Here is how to use this model to perform image and text matching

```python
from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
import requests
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")

# forward pass
scores = dict()
for text in texts:
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    scores[text] = outputs.logits[0,1].item()
```


Here is how to use this model to perform masked language modeling:

```python
from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000360943.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
text = "a <mask> looking out of the window"

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)

results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

print(results)
#.a cat looking out of the window.
```

## Training data

The BridgeTower model was pretrained on four public image-caption datasets:
- [Conceptual Captions (CC3M)](https://ai.google.com/research/ConceptualCaptions/)
- [Conceptual 12M (CC12M)](https://github.com/google-research-datasets/conceptual-12m)
- [SBU Captions](https://www.cs.rice.edu/~vo9/sbucaptions/)
- [MSCOCO Captions](https://arxiv.org/pdf/1504.00325.pdf)
- [Visual Genome](https://visualgenome.org/)
  
The total number of unique images in the combined data is around 14M. 

## Training procedure

### Pretraining

The model was pre-trained for 10 epochs on an Intel AI supercomputing cluster using 512 Gaudis and 128 Xeons with a batch size of 2048. 
The optimizer used was AdamW with a learning rate of 1e-7. No data augmentation was used except for center-crop. The image resolution in pre-training is set to 294 x 294. 

## Evaluation results
Please refer to [Table 5](https://arxiv.org/pdf/2206.08657.pdf) for BridgeTower's performance on Image Retrieval and other downstream tasks. 

### BibTeX entry and citation info
```bibtex
@article{xu2022bridge,
  title={BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning},
  author={Xu, Xiao and Wu, Chenfei and Rosenman, Shachar and Lal, Vasudev and Che, Wanxiang and Duan, Nan},
  journal={arXiv preprint arXiv:2206.08657},
  year={2022}
}
```