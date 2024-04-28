---
license: mit
tags:
- vision
- video-classification
---

# ViViT (Video Vision Transformer) 

ViViT model as introduced in the paper [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) by Arnab et al. and first released in [this repository](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit). 

Disclaimer: The team releasing ViViT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

ViViT is an extension of the [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/v4.27.0/model_doc/vit) to video.

We refer to the paper for details.

## Intended uses & limitations

The model is mostly meant to intended to be fine-tuned on a downstream task, like video classification. See the [model hub](https://huggingface.co/models?filter=vivit) to look for fine-tuned versions on a task that interests you.

### How to use

For code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/vivit).

### BibTeX entry and citation info

```bibtex
@misc{arnab2021vivit,
      title={ViViT: A Video Vision Transformer}, 
      author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
      year={2021},
      eprint={2103.15691},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```