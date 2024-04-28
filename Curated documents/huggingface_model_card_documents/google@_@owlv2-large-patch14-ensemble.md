---
license: apache-2.0
tags:
- vision
- zero-shot-object-detection
inference: false
---

# Model Card: OWLv2

## Model Details

The OWLv2 model (short for Open-World Localization) was proposed in [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) by Matthias Minderer, Alexey Gritsenko, Neil Houlsby. OWLv2, like OWL-ViT, is a zero-shot text-conditioned object detection model that can be used to query an image with one or multiple text queries.  

The model uses CLIP as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features. To use CLIP for detection, OWL-ViT removes the final token pooling layer of the vision model and attaches a lightweight classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the text model. The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads on standard detection datasets using a bipartite matching loss. One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection. 


### Model Date

June 2023

### Model Type

The model uses a CLIP backbone with a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss. The CLIP backbone is trained from scratch and fine-tuned together with the box and class prediction heads with an object detection objective.


### Documents

- [OWLv2 Paper](https://arxiv.org/abs/2306.09683)


### Use with Transformers

```python3
import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
```


## Model Use

### Intended Use

The model is intended as a research output for research communities. We hope that this model will enable researchers to better understand and explore zero-shot, text-conditioned object detection. We also hope it can be used for interdisciplinary studies of the potential impact of such models, especially in areas that commonly require identifying objects whose label is unavailable during training.

#### Primary intended uses

The primary intended users of these models are AI researchers.

We primarily imagine the model will be used by researchers to better understand robustness, generalization, and other capabilities, biases, and constraints of computer vision models.

## Data

The CLIP backbone of the model was trained on publicly available image-caption data. This was done through a combination of crawling a handful of websites and using commonly-used pre-existing image datasets such as [YFCC100M](http://projects.dfki.uni-kl.de/yfcc100m/). A large portion of the data comes from our crawling of the internet. This means that the data is more representative of people and societies most connected to the internet. The prediction heads of OWL-ViT, along with the CLIP backbone, are fine-tuned on publicly available object detection datasets such as [COCO](https://cocodataset.org/#home) and [OpenImages](https://storage.googleapis.com/openimages/web/index.html).

(to be updated for v2)

### BibTeX entry and citation info

```bibtex
@misc{minderer2023scaling,
      title={Scaling Open-Vocabulary Object Detection}, 
      author={Matthias Minderer and Alexey Gritsenko and Neil Houlsby},
      year={2023},
      eprint={2306.09683},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```