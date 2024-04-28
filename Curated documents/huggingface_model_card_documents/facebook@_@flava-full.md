---
license: bsd-3-clause
---
## Model Card: FLAVA

## Model Details

FLAVA model was developed by the researchers at FAIR to understand if a single model can work across different modalities with a unified architecture. The model was pretrained solely using publicly available multimodal datasets containing 70M image-text pairs in total and thus fully reproducible. Unimodal datasets ImageNet and BookCorpus + CCNews were also used to provide unimodal data to the model. The model (i) similar to CLIP can be used for arbitrary image classification tasks in a zero-shot manner (ii) used for image or text retrieval in a zero-shot manner (iii) can also be fine-tuned for natural language understanding (NLU) tasks such as GLUE and vision-and-language reasoning tasks such as VQA v2. The model is able to use the data available as images, text corpus and image-text pairs. In the original paper, the authors evaluate FLAVA on 32 tasks from computer vision, NLU and vision-and-language domains and show impressive performance across the board scoring higher micro-average than CLIP while being open.

## Model Date
Model was originally released in November 2021.

## Model Type

The FLAVA model uses a ViT-B/32 transformer for both image encoder and text encoder. FLAVA also employs a multimodal encoder on top for multimodal tasks such as vision-and-language tasks (VQA) which is a 6-layer encoder. Each component of FLAVA model can be loaded individually from `facebook/flava-full` checkpoint. If you need complete heads used for pretraining, please use `FlavaForPreTraining` model class otherwise `FlavaModel` should suffice for most use case. This [repository](https://github.com/facebookresearch/multimodal/tree/main/examples/flava) also contains code to pretrain the FLAVA model from scratch.

## Documents

- [FLAVA Paper](https://arxiv.org/abs/2112.04482)

## Using with Transformers

### FlavaModel

FLAVA model supports vision, language and multimodal inputs. You can pass inputs corresponding to the domain you are concerned with to get losses and outputs related to that domain.

```py
from PIL import Image
import requests

from transformers import FlavaProcessor, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
  text=["a photo of a cat", "a photo of a dog"], images=[image, image], return_tensors="pt", padding="max_length", max_length=77
)

outputs = model(**inputs)
image_embeddings = outputs.image_embeddings # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
text_embeddings = outputs.text_embeddings # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
multimodal_embeddings = outputs.multimodal_embeddings # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768
# Multimodal embeddings can be used for multimodal tasks such as VQA


## Pass only image
from transformers import FlavaFeatureExtractor

feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
inputs = feature_extractor(images=[image, image], return_tensors="pt")
outputs = model(**inputs)
image_embeddings = outputs.image_embeddings

## Pass only text
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding="max_length", max_length=77)
outputs = model(**inputs)
text_embeddings = outputs.text_embeddings
```

#### Encode Image

```py
from PIL import Image
import requests

from transformers import FlavaFeatureExtractor, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full")
feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=[image], return_tensors="pt")

image_embedding = model.get_image_features(**inputs)
```

#### Encode Text

```py
from PIL import Image

from transformers import BertTokenizer, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full")
tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

inputs = tokenizer(text=["a photo of a dog"], return_tensors="pt", padding="max_length", max_length=77)

text_embedding = model.get_text_features(**inputs)
```

### FlavaForPreTraining

FLAVA model supports vision, language and multimodal inputs. You can pass corresponding inputs to modality to get losses and outputs related to that domain.

```py
from PIL import Image
import requests

from transformers import FlavaProcessor, FlavaForPreTraining

model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
  text=["a photo of a cat", "a photo of a dog"], 
  images=[image, image], 
  return_tensors="pt", 
  padding="max_length", 
  max_length=77,
  return_codebook_pixels=True,
  return_image_mask=True,
  # Other things such as mlm_labels, itm_labels can be passed here. See docs
)
inputs.bool_masked_pos.zero_()

outputs = model(**inputs)
image_embeddings = outputs.image_embeddings # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
text_embeddings = outputs.text_embeddings # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
# Multimodal embeddings can be used for multimodal tasks such as VQA
multimodal_embeddings = outputs.multimodal_embeddings # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768

# Loss
loss = output.loss # probably NaN due to missing labels

# Global contrastive loss logits
image_contrastive_logits = outputs.contrastive_logits_per_image
text_contrastive_logits = outputs.contrastive_logits_per_text

# ITM logits
itm_logits = outputs.itm_logits

```

### FlavaImageModel

```py
from PIL import Image
import requests

from transformers import FlavaFeatureExtractor, FlavaImageModel

model = FlavaImageModel.from_pretrained("facebook/flava-full")
feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=[image], return_tensors="pt")

outputs = model(**inputs)
image_embeddings = outputs.last_hidden_state
```

### FlavaTextModel

```py
from PIL import Image

from transformers import BertTokenizer, FlavaTextModel

model = FlavaTextModel.from_pretrained("facebook/flava-full")
tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

inputs = tokenizer(text=["a photo of a dog"], return_tensors="pt", padding="max_length", max_length=77)

outputs = model(**inputs)
text_embeddings = outputs.last_hidden_state
```

## Model Use

## Intended Use
The model is intended to serve as a reproducible research artifact for research communities in the light of models whose exact reproduction details are never released such as [CLIP](https://github.com/openai/CLIP) and [SimVLM](https://arxiv.org/abs/2108.10904). FLAVA model performs equivalently to these models on most tasks while being trained on less (70M pairs compared to CLIP's 400M and SimVLM's 1.8B pairs respectively) but public data. We hope that this model enable communities to better understand, and explore zero-shot and arbitrary image classification, multi-domain pretraining, modality-agnostic generic architectures while also providing a chance to develop on top of it. 

## Primary Intended Uses

The primary intended users of these models are AI researchers.

We primarily imagine the model will be used by researchers to better understand robustness, generalization, and other capabilities, biases, and constraints of foundation models which work across domains which in this case are vision, language and combined multimodal vision-and-language domain.

## Out-of-Scope Use Cases

Similar to CLIP, **Any** deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed use cases such as image search in a constrained environment, are also not recommended unless there is thorough in-domain testing of the model with a specific, fixed class taxonomy. Though FLAVA is trained on open and public data which doesn't contain a lot of harmful data, users should still employ proper safety measures.

Certain use cases which would fall under the domain of surveillance and facial recognition are always out-of-scope regardless of performance of the model. This is because the use of artificial intelligence for tasks such as these can be premature currently given the lack of testing norms and checks to ensure its fair use.

Since the model has not been purposefully trained in or evaluated on any languages other than English, its use should be limited to English language use cases.

## Data

FLAVA was pretrained on public available 70M image and text pairs. This includes datasets such as COCO, Visual Genome, Localized Narratives, RedCaps, a custom filtered subset of YFCC100M, SBUCaptions, Conceptual Captions and Wikipedia Image-Text datasets. A larger portion of this dataset comes from internet and thus can have bias towards people most connected to internet such as those from developed countries and younger, male users. 

## Data Mission Statement
Our goal with building this dataset called PMD (Public Multimodal Datasets) was two-fold (i) allow reproducibility of vision-language foundation models with publicly available data and (ii) test robustness and generalizability of FLAVA across the domains. The data was collected from already existing public dataset sources which have already been filtered out by the original dataset curators to not contain adult and excessively violent content. We will make the URLs of the images public for further research reproducibility.

## Performance and Limitations
## Performance

FLAVA has been evaluated on 35 different tasks from computer vision, natural language understanding, and vision-and-language reasoning. 
On COCO and Flickr30k retrieval, we report zero-shot accuracy, on image tasks, we report linear-eval and on rest of the tasks, we report fine-tuned accuracies. Generally, FLAVA works much better than CLIP where tasks require good text understanding. The paper describes more in details but following are the 35 datasets:

### Natural Language Understanding
- MNLI
- CoLA
- MRPC
- QQP
- SST-2
- QNLI
- RTE
- STS-B

### Image Understanding

- ImageNet
- Food100
- CIFAR10
- CIFAR100
- Cars
- Aircraft
- DTD
- Pets
- Caltech101
- Flowers102
- MNIST
- STL10
- EuroSAT
- GTSRB
- KITTI
- PCAM
- UCF101
- CLEVR
- FER 2013
- SUN397
- Image SST
- Country 211

### Vision and Language Reasoning
- VQA v2
- SNLI-VE
- Hateful Memes
- Flickr30K Retrieval
- COCO Retrieval

## Limitations

Currently, FLAVA has many limitations. The image classification accuracy is not on par with CLIP on some of the tasks while text accuracy is not on par with BERT on some of the tasks suggesting possible room for improvement. FLAVA also doesn't work well on tasks containing scene text given the lack of scene text in most public datasets. Additionally, similar to CLIP, our approach to testing FLAVA also has an important limitation in the case of image tasks, where we use linear probes to evaluate FLAVA and there is evidence suggesting that linear probes can underestimate model performance.

## Feedback/Questions

Please email Amanpreet at `amanpreet [at] nyu [dot] edu` for questions.
