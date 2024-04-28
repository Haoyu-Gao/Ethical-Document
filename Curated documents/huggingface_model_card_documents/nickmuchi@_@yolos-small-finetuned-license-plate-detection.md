---
language:
- en
tags:
- object-detection
- license-plate-detection
- vehicle-detection
metrics:
- average precision
- recall
- IOU
widget:
- src: https://drive.google.com/uc?id=1j9VZQ4NDS4gsubFf3m2qQoTMWLk552bQ
  example_title: Skoda 1
- src: https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5
  example_title: Skoda 2
pipeline_tag: object-detection
---
# YOLOS (small-sized) model
This model is a fine-tuned version of [hustvl/yolos-small](https://huggingface.co/hustvl/yolos-small) on the [licesne-plate-recognition](https://app.roboflow.com/objectdetection-jhgr1/license-plates-recognition/2) dataset from Roboflow which contains 5200 images in the training set and 380 in the validation set.
The original YOLOS model was fine-tuned on COCO 2017 object detection (118k annotated images).

## Model description

YOLOS is a Vision Transformer (ViT) trained using the DETR loss. Despite its simplicity, a base-sized YOLOS model is able to achieve 42 AP on COCO validation 2017 (similar to DETR and more complex frameworks such as Faster R-CNN).
## Intended uses & limitations
You can use the raw model for object detection. See the [model hub](https://huggingface.co/models?search=hustvl/yolos) to look for all available YOLOS models.

### How to use

Here is how to use this model:

```python
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

url = 'https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding face mask detection classes
logits = outputs.logits
bboxes = outputs.pred_boxes
```
Currently, both the feature extractor and model support PyTorch.

## Training data

The YOLOS model was pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet2012) and fine-tuned on [COCO 2017 object detection](https://cocodataset.org/#download), a dataset consisting of 118k/5k annotated images for training/validation respectively. 

### Training

This model was fine-tuned for 200 epochs on the [licesne-plate-recognition](https://app.roboflow.com/objectdetection-jhgr1/license-plates-recognition/2).

## Evaluation results

This model achieves an AP (average precision) of **49.0**.

Accumulating evaluation results...

IoU metric: bbox

Metrics           | Metric Parameter      | Location    | Dets          | Value |
----------------  | --------------------- | ------------| ------------- | ----- |
Average Precision | (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] | 0.490 |
Average Precision | (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] | 0.792 |
Average Precision | (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] | 0.585 |
Average Precision | (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] | 0.167 |
Average Precision | (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] | 0.460 |
Average Precision | (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] | 0.824 |
Average Recall    | (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] | 0.447 |
Average Recall    | (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] | 0.671 |
Average Recall    | (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] | 0.676 |
Average Recall    | (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] | 0.278 |
Average Recall    | (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] | 0.641 |
Average Recall    | (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] | 0.890 |