---
license: apache-2.0
tags:
- image-classification
datasets:
- imagenet
---

# Data-efficient Image Transformer (tiny-sized model) 

Data-efficient Image Transformer (DeiT) model pre-trained and fine-tuned on ImageNet-1k (1 million images, 1,000 classes) at resolution 224x224. It was first introduced in the paper [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) by Touvron et al. and first released in [this repository](https://github.com/facebookresearch/deit). However, the weights were converted from the [timm repository](https://github.com/rwightman/pytorch-image-models) by Ross Wightman.  

Disclaimer: The team releasing DeiT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

This model is actually a more efficiently trained Vision Transformer (ViT).

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pre-trained and fine-tuned on a large collection of images in a supervised fashion, namely ImageNet-1k, at a resolution of 224x224 pixels. 

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=facebook/deit) to look for
fine-tuned versions on a task that interests you.

### How to use

Since this model is a more efficiently trained ViT model, you can plug it into ViTModel or ViTForImageClassification. Note that the model expects the data to be prepared using DeiTFeatureExtractor. Here we use AutoFeatureExtractor, which will automatically use the appropriate feature extractor given the model name. 

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import AutoFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-patch16-224')
model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

Currently, both the feature extractor and model support PyTorch. Tensorflow and JAX/FLAX are coming soon.

## Training data

The ViT model was pretrained on [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/), a dataset consisting of 1 million images and 1k classes. 

## Training procedure

### Preprocessing

The exact details of preprocessing of images during training/validation can be found [here](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py#L78). 

At inference time, images are resized/rescaled to the same resolution (256x256), center-cropped at 224x224 and normalized across the RGB channels with the ImageNet mean and standard deviation.

### Pretraining

The model was trained on a single 8-GPU node for 3 days. Training resolution is 224. For all hyperparameters (such as batch size and learning rate) we refer to table 9 of the original paper.

## Evaluation results

| Model                                 | ImageNet top-1 accuracy | ImageNet top-5 accuracy | # params | URL                                                              |
|---------------------------------------|-------------------------|-------------------------|----------|------------------------------------------------------------------|
| **DeiT-tiny**                         | **72.2**                | **91.1**                | **5M**   | **https://huggingface.co/facebook/deit-tiny-patch16-224**            |
| DeiT-small                            | 79.9                    | 95.0                    | 22M      | https://huggingface.co/facebook/deit-small-patch16-224           |
| DeiT-base                             | 81.8                    | 95.6                    | 86M      | https://huggingface.co/facebook/deit-base-patch16-224            |
| DeiT-tiny distilled                   | 74.5                    | 91.9                    | 6M       | https://huggingface.co/facebook/deit-tiny-distilled-patch16-224  |
| DeiT-small distilled                  | 81.2                    | 95.4                    | 22M      | https://huggingface.co/facebook/deit-small-distilled-patch16-224 |
| DeiT-base distilled                   | 83.4                    | 96.5                    | 87M      | https://huggingface.co/facebook/deit-base-distilled-patch16-224  |
| DeiT-base 384                         | 82.9                    | 96.2                    | 87M      | https://huggingface.co/facebook/deit-base-patch16-384            |
| DeiT-base distilled 384 (1000 epochs) | 85.2                    | 97.2                    | 88M      | https://huggingface.co/facebook/deit-base-distilled-patch16-384  |

Note that for fine-tuning, the best results are obtained with a higher resolution (384x384). Of course, increasing the model size will result in better performance.

### BibTeX entry and citation info

```bibtex
@misc{touvron2021training,
      title={Training data-efficient image transformers & distillation through attention}, 
      author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
      year={2021},
      eprint={2012.12877},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={Ieee}
}
```