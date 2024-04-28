---
tags:
- image-classification
- pytorch
datasets:
- fairface
---



A vision transformer finetuned to classify the age of a given person's face. 




```python
import requests
from PIL import Image
from io import BytesIO

from transformers import ViTFeatureExtractor, ViTForImageClassification

# Get example image from official fairface repo + read it in as an image
r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
im = Image.open(BytesIO(r.content))

# Init model, transforms
model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

# Transform our image and pass it through the model
inputs = transforms(im, return_tensors='pt')
output = model(**inputs)

# Predicted Class probabilities
proba = output.logits.softmax(1)

# Predicted Classes
preds = proba.argmax(1)
```