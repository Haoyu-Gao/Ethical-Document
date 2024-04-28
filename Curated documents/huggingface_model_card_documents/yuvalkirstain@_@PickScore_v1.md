---
{}
---
# Model Card for PickScore v1

This model is a scoring function for images generated from text. It takes as input a prompt and a generated image and outputs a score. 
It can be used as a general scoring function, and for tasks such as human preference prediction, model evaluation, image ranking, and more. 
See our paper [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569) for more details.


## Model Details

### Model Description

This model was finetuned from CLIP-H using the [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1).

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [See the PickScore repo](https://github.com/yuvalkirstain/PickScore)
- **Paper:** [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569).
- **Demo [optional]:** [Huggingface Spaces demo for PickScore](https://huggingface.co/spaces/yuvalkirstain/PickScore)

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# import
from transformers import AutoProcessor, AutoModel

# load model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

pil_images = [Image.open("my_amazing_images/1.jpg"), Image.open("my_amazing_images/2.jpg")]
prompt = "fantastic, increadible prompt"
print(calc_probs(prompt, pil_images))
```
## Training Details

### Training Data

This model was trained on the [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1).


### Training Procedure 

TODO - add paper.


## Citation [optional]

If you find this work useful, please cite:

```bibtex
@inproceedings{Kirstain2023PickaPicAO,
  title={Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation},
  author={Yuval Kirstain and Adam Polyak and Uriel Singer and Shahbuland Matiana and Joe Penna and Omer Levy},
  year={2023}
}
```

**APA:**

[More Information Needed]


