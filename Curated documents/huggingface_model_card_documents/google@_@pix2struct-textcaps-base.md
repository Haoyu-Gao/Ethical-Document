---
language:
- en
- fr
- ro
- de
- multilingual
license: apache-2.0
pipeline_tag: image-to-text
inference: false
---


# Model card for Pix2Struct - Finetuned on TextCaps

![model_image](https://s3.amazonaws.com/moonup/production/uploads/1678713353867-62441d1d9fdefb55a0b7d12c.png)

#  Table of Contents

0. [TL;DR](#TL;DR)
1. [Using the model](#using-the-model)
2. [Contribution](#contribution)
3. [Citation](#citation)

# TL;DR

Pix2Struct is an image encoder - text decoder model that is trained on image-text pairs for various tasks, including image captionning and visual question answering. The full list of available models can be found on the Table 1 of the paper:

![Table 1 - paper](https://s3.amazonaws.com/moonup/production/uploads/1678712985040-62441d1d9fdefb55a0b7d12c.png)


The abstract of the model states that: 
> Visually-situated language is ubiquitous—sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and
forms. Perhaps due to this diversity, previous work has typically relied on domainspecific recipes with limited sharing of the underlying data, model architectures,
and objectives. We present Pix2Struct, a pretrained image-to-text model for
purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse
masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large
source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, image captioning. In addition to the novel pretraining strategy,
we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions
are rendered directly on top of the input image. For the first time, we show that a
single pretrained model can achieve state-of-the-art results in six out of nine tasks
across four domains: documents, illustrations, user interfaces, and natural images.

# Using the model 

## Converting from T5x to huggingface

You can use the [`convert_pix2struct_checkpoint_to_pytorch.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/convert_pix2struct_checkpoint_to_pytorch.py) script as follows:
```bash
python convert_pix2struct_checkpoint_to_pytorch.py --t5x_checkpoint_path PATH_TO_T5X_CHECKPOINTS --pytorch_dump_path PATH_TO_SAVE
```
if you are converting a large model, run:
```bash
python convert_pix2struct_checkpoint_to_pytorch.py --t5x_checkpoint_path PATH_TO_T5X_CHECKPOINTS --pytorch_dump_path PATH_TO_SAVE --use-large
```
Once saved, you can push your converted model with the following snippet:
```python
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

model = Pix2StructForConditionalGeneration.from_pretrained(PATH_TO_SAVE)
processor = Pix2StructProcessor.from_pretrained(PATH_TO_SAVE)

model.push_to_hub("USERNAME/MODEL_NAME")
processor.push_to_hub("USERNAME/MODEL_NAME")
```

## Running the model

### In full precision, on CPU:

You can run the model in full precision on CPU:
```python
import requests
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, return_tensors="pt")

predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))
>>> A stop sign is on a street corner.
```

### In full precision, on GPU:

You can run the model in full precision on CPU:
```python
import requests
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base").to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, return_tensors="pt").to("cuda")

predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))
>>> A stop sign is on a street corner.
```

### In half precision, on GPU:

You can run the model in full precision on CPU:
```python
import requests
import torch

from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base", torch_dtype=torch.bfloat16).to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)

predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))
>>> A stop sign is on a street corner.
```

### Use different sequence length

This model has been trained on a sequence length of `2048`. You can try to reduce the sequence length for a more memory efficient inference but you may observe some performance degradation for small sequence length (<512). Just pass `max_patches` when calling the processor:
```python
inputs = processor(images=image, return_tensors="pt", max_patches=512)
```

### Conditional generation

You can also pre-pend some input text to perform conditional generation:

```python
import requests
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "A picture of"

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, text=text, return_tensors="pt")

predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))
>>> A picture of a stop sign that says yes.
```

# Contribution

This model was originally contributed by Kenton Lee, Mandar Joshi et al. and added to the Hugging Face ecosystem by [Younes Belkada](https://huggingface.co/ybelkada).

# Citation

If you want to cite this work, please consider citing the original paper:
```
@misc{https://doi.org/10.48550/arxiv.2210.03347,
  doi = {10.48550/ARXIV.2210.03347},
  
  url = {https://arxiv.org/abs/2210.03347},
  
  author = {Lee, Kenton and Joshi, Mandar and Turc, Iulia and Hu, Hexiang and Liu, Fangyu and Eisenschlos, Julian and Khandelwal, Urvashi and Shaw, Peter and Chang, Ming-Wei and Toutanova, Kristina},
  
  keywords = {Computation and Language (cs.CL), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```