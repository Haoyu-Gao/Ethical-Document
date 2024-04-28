---
language: en
tags:
- BigBird
- clinical
---

<span style="font-size:larger;">**Clinical-BigBird**</span> is a clinical knowledge enriched version of BigBird that was further pre-trained using MIMIC-III clinical notes. It allows up to 4,096 tokens as the model input. Clinical-BigBird consistently out-performs ClinicalBERT across 10 baseline dataset. Those downstream experiments broadly cover named entity recognition (NER), question answering (QA), natural language inference (NLI) and text classification tasks. For more details, please refer to [our paper](https://arxiv.org/pdf/2201.11838.pdf).
We also provide a sister model at [Clinical-Longformer](https://huggingface.co/yikuan8/Clinical-Longformer)


### Pre-training
We initialized Clinical-BigBird from the pre-trained weights of the base version of BigBird. The pre-training process was distributed in parallel to 6 32GB Tesla V100 GPUs. FP16 precision was enabled to accelerate training. We pre-trained Clinical-BigBird for 300,000 steps with batch size of 6×2. The learning rates were 3e-5. The entire pre-training process took more than 2 weeks. 


### Usage
Load the model directly from Transformers:
```
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird")
```
### Citing
If you find our model helps, please consider citing this :)
```
@article{li2022clinical,
  title={Clinical-Longformer and Clinical-BigBird: Transformers for long clinical sequences},
  author={Li, Yikuan and Wehbe, Ramsey M and Ahmad, Faraz S and Wang, Hanyin and Luo, Yuan},
  journal={arXiv preprint arXiv:2201.11838},
  year={2022}
}
```

### Questions
Please email yikuanli2018@u.northwestern.edu



