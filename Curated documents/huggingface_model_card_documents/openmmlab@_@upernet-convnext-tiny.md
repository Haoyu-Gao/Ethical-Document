---
language: en
license: mit
tags:
- vision
- image-segmentation
model_name: openmmlab/upernet-convnext-tiny
---

# UperNet, ConvNeXt tiny-sized backbone

UperNet framework for semantic segmentation, leveraging a ConvNeXt backbone. UperNet was introduced in the paper [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) by Xiao et al.

Combining UperNet with a ConvNeXt backbone was introduced in the paper [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).

Disclaimer: The team releasing UperNet + ConvNeXt did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

UperNet is a framework for semantic segmentation. It consists of several components, including a backbone, a Feature Pyramid Network (FPN) and a Pyramid Pooling Module (PPM).

Any visual backbone can be plugged into the UperNet framework. The framework predicts a semantic label per pixel.

![UperNet architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/upernet_architecture.jpg)

## Intended uses & limitations

You can use the raw model for semantic segmentation. See the [model hub](https://huggingface.co/models?search=openmmlab/upernet) to look for
fine-tuned versions (with various backbones) on a task that interests you.

### How to use

For code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation).
