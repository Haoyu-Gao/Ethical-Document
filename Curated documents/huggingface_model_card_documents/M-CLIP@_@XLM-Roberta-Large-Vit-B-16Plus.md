---
language:
- multilingual
- af
- sq
- am
- ar
- az
- bn
- bs
- bg
- ca
- zh
- hr
- cs
- da
- nl
- en
- et
- fr
- de
- el
- hi
- hu
- is
- id
- it
- ja
- mk
- ml
- mr
- pl
- pt
- ro
- ru
- sr
- sl
- es
- sw
- sv
- tl
- te
- tr
- tk
- uk
- ur
- ug
- uz
- vi
- xh
---

## Multilingual-clip: XLM-Roberta-Large-Vit-B-16Plus

Multilingual-CLIP extends OpenAI's English text encoders to multiple other languages. This model *only* contains the multilingual text encoder. The corresponding image model `Vit-B-16Plus` can be retrieved via instructions found on `mlfoundations` [open_clip repository on Github](https://github.com/mlfoundations/open_clip). We provide a usage example below. 

## Requirements

To use both the multilingual text encoder and corresponding image encoder, we need to install the packages [`multilingual-clip`](https://github.com/FreddeFrallan/Multilingual-CLIP) and [`open_clip_torch`](https://github.com/mlfoundations/open_clip). 

```
pip install multilingual-clip
pip install open_clip_torch
```

## Usage

Extracting embeddings from the text encoder can be done in the following way:

```python
from multilingual_clip import pt_multilingual_clip
import transformers

texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
]
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'

# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

embeddings = model.forward(texts, tokenizer)
print("Text features shape:", embeddings.shape)
```

Extracting embeddings from the corresponding image encoder:

```python
import torch
import open_clip
import requests
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)

print("Image features shape:", image_features.shape) 
```

## Evaluation results

None of the M-CLIP models have been extensivly evaluated, but testing them on Txt2Img retrieval on the humanly translated MS-COCO dataset, we see the following **R@10** results:

| Name | En | De | Es | Fr | Zh | It | Pl | Ko | Ru | Tr | Jp |
| ----------------------------------|:-----: |:-----: |:-----: |:-----: | :-----: |:-----: |:-----: |:-----: |:-----: |:-----: |:-----: |
| [OpenAI CLIP Vit-B/32](https://github.com/openai/CLIP)| 90.3 | - | - | - | - | - | - | - | - | - | - |
| [OpenAI CLIP Vit-L/14](https://github.com/openai/CLIP)| 91.8 | - | - | - | - | - | - | - | - | - | - |
| [OpenCLIP ViT-B-16+-](https://github.com/openai/CLIP)| 94.3 | - | - | - | - | - | - | - | - | - | - |
| [LABSE Vit-L/14](https://huggingface.co/M-CLIP/LABSE-Vit-L-14)| 91.6 | 89.6 | 89.5 | 89.9 | 88.9 | 90.1 | 89.8 | 80.8 | 85.5 | 89.8 | 73.9 |
| [XLM-R Large Vit-B/32](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-32)| 91.8 | 88.7 | 89.1 | 89.4 | 89.3 | 89.8| 91.4 | 82.1 | 86.1 | 88.8 | 81.0 |
| [XLM-R Vit-L/14](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14)| 92.4 | 90.6 | 91.0 | 90.0 | 89.7 | 91.1 | 91.3 | 85.2 | 85.8 | 90.3 | 81.9 |
| [XLM-R Large Vit-B/16+](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus)| **95.0** | **93.0** | **93.6** | **93.1** | **94.0** | **93.1** | **94.4** | **89.0** | **90.0** | **93.0** | **84.2** |


## Training/Model details

Further details about the model training and data can be found in the [model card](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/larger_mclip.md).