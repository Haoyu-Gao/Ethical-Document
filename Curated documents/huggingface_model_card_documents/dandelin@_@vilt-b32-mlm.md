---
license: apache-2.0
---

# Vision-and-Language Transformer (ViLT), pre-trained only

Vision-and-Language Transformer (ViLT) model pre-trained on GCC+SBU+COCO+VG (200k steps). It was introduced in the paper [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) by Kim et al. and first released in [this repository](https://github.com/dandelin/ViLT). Note: this model only includes the language modeling head.

Disclaimer: The team releasing ViLT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Intended uses & limitations

You can use the raw model for masked language modeling given an image and a piece of text with [MASK] tokens.

### How to use

Here is how to use this model in PyTorch:

```
from transformers import ViltProcessor, ViltForMaskedLM
import requests
from PIL import Image
import re

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "a bunch of [MASK] laying on a [MASK]."

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)

tl = len(re.findall("\[MASK\]", text))
inferred_token = [text]

# gradually fill in the MASK tokens, one by one
with torch.no_grad():
    for i in range(tl):
        encoded = processor.tokenizer(inferred_token)
        input_ids = torch.tensor(encoded.input_ids).to(device)
        encoded = encoded["input_ids"][0][1:-1]
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        mlm_logits = outputs.logits[0]  # shape (seq_len, vocab_size)
        # only take into account text features (minus CLS and SEP token)
        mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
        mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
        # only take into account text
        mlm_values[torch.tensor(encoded) != 103] = 0
        select = mlm_values.argmax().item()
        encoded[select] = mlm_ids[select].item()
        inferred_token = [processor.decode(encoded)]

selected_token = ""
encoded = processor.tokenizer(inferred_token)
processor.decode(encoded.input_ids[0], skip_special_tokens=True)
```

## Training data

(to do)

## Training procedure

### Preprocessing

(to do)

### Pretraining

(to do)

## Evaluation results

(to do)

### BibTeX entry and citation info

```bibtex
@misc{kim2021vilt,
      title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}, 
      author={Wonjae Kim and Bokyung Son and Ildoo Kim},
      year={2021},
      eprint={2102.03334},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```