---
license: cc-by-nc-4.0
tags:
- vision
- video-classification
---

# TimeSformer (base-sized model, fine-tuned on Kinetics-400) 

TimeSformer model pre-trained on [Kinetics-400](https://www.deepmind.com/open-source/kinetics). It was introduced in the paper [TimeSformer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) by Tong et al. and first released in [this repository](https://github.com/facebookresearch/TimeSformer). 

Disclaimer: The team releasing TimeSformer did not write a model card for this model so this model card has been written by [fcakyon](https://github.com/fcakyon).

## Intended uses & limitations

You can use the raw model for video classification into one of the 400 possible Kinetics-400 labels.

### How to use

Here is how to use this model to classify a video:

```python
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

video = list(np.random.randn(8, 3, 224, 224))

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

inputs = processor(video, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

For more code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/timesformer.html#).

### BibTeX entry and citation info

```bibtex
@inproceedings{bertasius2021space,
  title={Is Space-Time Attention All You Need for Video Understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={International Conference on Machine Learning},
  pages={813--824},
  year={2021},
  organization={PMLR}
}
```