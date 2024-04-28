---
library_name: ultralytics
tags:
- ultralyticsplus
- yolov8
- ultralytics
- yolo
- vision
- object-detection
- pytorch
- awesome-yolov8-models
datasets:
- keremberke/table-extraction
library_version: 8.0.21
inference: false
model-index:
- name: keremberke/yolov8s-table-extraction
  results:
  - task:
      type: object-detection
    dataset:
      name: table-extraction
      type: keremberke/table-extraction
      split: validation
    metrics:
    - type: precision
      value: 0.98376
      name: mAP@0.5(box)
---

<div align="center">
  <img width="640" alt="keremberke/yolov8s-table-extraction" src="https://huggingface.co/keremberke/yolov8s-table-extraction/resolve/main/thumbnail.jpg">
</div>

### Supported Labels

```
['bordered', 'borderless']
```

### How to use

- Install [ultralyticsplus](https://github.com/fcakyon/ultralyticsplus):

```bash
pip install ultralyticsplus==0.0.23 ultralytics==8.0.21
```

- Load model and perform prediction:

```python
from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('keremberke/yolov8s-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model.predict(image)

# observe results
print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()
```

**More models available at: [awesome-yolov8-models](https://yolov8.xyz)**