---
tags:
- mgp-str
- image-to-text
widget:
- src: https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/OCR/MGP-STR/demo_imgs/IIIT5k_HOUSE.png
  example_title: Example 1
- src: https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/OCR/MGP-STR/demo_imgs/IIT5k_EVERYONE.png
  example_title: Example 2
- src: https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/OCR/MGP-STR/demo_imgs/CUTE80_KINGDOM.png
  example_title: Example 3
---

# MGP-STR (base-sized model) 

MGP-STR base-sized model is trained on MJSynth and SynthText. It was introduced in the paper [Multi-Granularity Prediction for Scene Text Recognition](https://arxiv.org/abs/2209.03592) and first released in [this repository](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR). 

## Model description

MGP-STR is pure vision STR model, consisting of ViT and specially designed A^3 modules. The ViT module was initialized from the weights of DeiT-base, except the patch embedding model, due to the inconsistent input size.

Images (32x128) are presented to the model as a sequence of fixed-size patches (resolution 4x4), which are linearly embedded. One also adds absolute position embeddings before feeding the sequence to the layers of the ViT module. Next, A^3 module selects a meaningful combination from the tokens of ViT output and integrates them into one output token corresponding to a specific character. Moreover, subword classification heads based on BPE A^3 module and WordPiece A^3 module are devised for subword predictions, so that the language information can be implicitly modeled. Finally, these multi-granularity predictions (character, subword and even word) are merged via a simple and effective fusion strategy.

## Intended uses & limitations

You can use the raw model for optical character recognition (OCR) on text images. See the [model hub](https://huggingface.co/models?search=alibaba-damo/mgp-str) to look for fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model in PyTorch:

```python
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
import requests
from PIL import Image

processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

# load image from the IIIT-5k dataset
url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
outputs = model(pixel_values)

generated_text = processor.batch_decode(outputs.logits)['generated_text']
```

### BibTeX entry and citation info

```bibtex
@inproceedings{ECCV2022mgp_str,
  title={Multi-Granularity Prediction for Scene Text Recognition},
  author={Peng Wang, Cheng Da, and Cong Yao},
  booktitle = {ECCV},
  year={2022}
}
```