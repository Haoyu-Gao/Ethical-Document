---
language:
- en
license: cc-by-sa-4.0
tags:
- Multimodal
- StableLM
datasets:
- LDJnr/LessWrong-Amplify-Instruct
- LDJnr/Pure-Dove
- LDJnr/Verified-Camel
---

# Obsidian: Worlds smallest multi-modal LLM. First multi-modal model in size 3B

## Model Name: Obsidian-3B-V0.5

Obsidian is a brand new series of Multimodal Language Models. This first project is led by Quan N. and Luigi D.(LDJ).

Obsidian-3B-V0.5 is a multi-modal AI model that has vision! it's smarts are built on [Capybara-3B-V1.9](https://huggingface.co/NousResearch/Capybara-3B-V1.9) based on [StableLM-3B-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t). Capybara-3B-V1.9 achieves state-of-the-art performance when compared to model with similar size, even beats some 7B models.

Current finetuning and inference code is available on our GitHub repo: [Here](https://github.com/NousResearch/Obsidian)

## Acknowledgement

Obsidian-3B-V0.5 was developed and finetuned by [Nous Research](https://huggingface.co/NousResearch), in collaboration with [Virtual Interactive](https://huggingface.co/vilm).
Special thank you to **LDJ** for the wonderful Capybara dataset, and **qnguyen3** for the model training procedure.
## Model Training

Obsidian-3B-V0.5 followed the same training procedure as LLaVA 1.5

## Prompt Format

The model followed ChatML format. However, with `###` as the seperator

```
<|im_start|>user
What is this sign about?\n<image>
###
<|im_start|>assistant
The sign is about bullying, and it is placed on a black background with a red background.
###
```

## Benchmarks

Coming Soon!


Citation:

```
@article{nguyen2023Obsidian-3B,
  title={Obsidian-3B: First Multi-modal below 7B Parameters.},
  author={Nguyen, Quan and Daniele},
  journal={HuggingFace:https://huggingface.co/NousResearch/Obsidian-3B-V0.5},
  year={2023}
}
```

Acknowledgements:

```
@article{daniele2023amplify-instruct,
  title={Amplify-Instruct: Synthetically Generated Diverse Multi-turn Conversations for Effecient LLM Training.},
  author={Daniele, Luigi and Suphavadeeprasit},
  journal={arXiv preprint arXiv:(comming soon)},
  year={2023}
}
```