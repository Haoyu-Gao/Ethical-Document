---
language: ja
license: apache-2.0
tags:
- feature-extraction
- ja
- japanese
- clip
- vision
thumbnail: https://github.com/rinnakk/japanese-pretrained-models/blob/master/rinna.png
---

# rinna/japanese-clip-vit-b-16

![rinna-icon](./rinna.png)

This is a Japanese [CLIP (Contrastive Language-Image Pre-Training)](https://arxiv.org/abs/2103.00020) model trained by [rinna Co., Ltd.](https://corp.rinna.co.jp/).

Please see [japanese-clip](https://github.com/rinnakk/japanese-clip) for the other available models.


# How to use the model


1. Install package

```shell
$ pip install git+https://github.com/rinnakk/japanese-clip.git
```

2. Run

```python
import io
import requests
from PIL import Image
import torch
import japanese_clip as ja_clip

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
tokenizer = ja_clip.load_tokenizer()

img = Image.open(io.BytesIO(requests.get('https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260').content))
image = preprocess(img).unsqueeze(0).to(device)
encodings = ja_clip.tokenize(
    texts=["犬", "猫", "象"],
    max_seq_len=77,
    device=device,
    tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(**encodings)
    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1.0, 0.0, 0.0]]
```

# Model architecture
The model was trained  a ViT-B/16 Transformer architecture as an image encoder and uses a 12-layer BERT as a text encoder. The image encoder was initialized from the [AugReg `vit-base-patch16-224` model](https://github.com/google-research/vision_transformer).

# Training
The model was trained on [CC12M](https://github.com/google-research-datasets/conceptual-12m) translated the captions to Japanese.


# License

[The Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
