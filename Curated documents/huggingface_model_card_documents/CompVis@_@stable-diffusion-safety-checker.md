---
tags:
- clip
---

# Model Card for stable-diffusion-safety-checker
 
# Model Details
 
## Model Description
 
More information needed
 
- **Developed by:** More information needed
- **Shared by [Optional]:** CompVis
- **Model type:** Image Identification 
- **Language(s) (NLP):** More information needed
- **License:** More information needed
- **Parent Model:** [CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- **Resources for more information:** 
	- [CLIP Paper](https://arxiv.org/abs/2103.00020)
	- [Stable Diffusion Model Card](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md)


# Uses
 

## Direct Use
This model can be used for identifying NSFW image 

The CLIP model devlopers note in their [model card](https://huggingface.co/openai/clip-vit-large-patch14) : 

>The primary intended users of these models are AI researchers.

We primarily imagine the model will be used by researchers to better understand robustness, generalization, and other capabilities, biases, and constraints of computer vision models.


 
## Downstream Use [Optional]
 
More information needed.
 
## Out-of-Scope Use
 
The model is not intended to be used with transformers but with diffusers. This model should also not be used to intentionally create hostile or alienating environments for people. 
 
# Bias, Risks, and Limitations
 
 
Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). Predictions generated by the model may include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups.

The CLIP model devlopers note in their [model card](https://huggingface.co/openai/clip-vit-large-patch14) : 
> We find that the performance of CLIP - and the specific biases it exhibits - can depend significantly on class design and the choices one makes for categories to include and exclude. We tested the risk of certain kinds of denigration with CLIP by classifying images of people from Fairface into crime-related and non-human animal categories. We found significant disparities with respect to race and gender. Additionally, we found that these disparities could shift based on how the classes were constructed. 

> We also tested the performance of CLIP on gender, race and age classification using the Fairface dataset (We default to using race categories as they are constructed in the Fairface dataset.) in order to assess quality of performance across different demographics. We found accuracy >96% across all races for gender classification with ‘Middle Eastern’ having the highest accuracy (98.4%) and ‘White’ having the lowest (96.5%). Additionally, CLIP averaged ~93% for racial classification and ~63% for age classification

## Recommendations
 
 
Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

# Training Details
 
## Training Data
 
More information needed 
 
## Training Procedure

 
### Preprocessing
 
More information needed 


 
### Speeds, Sizes, Times
 
More information needed 


 
# Evaluation
 
 
## Testing Data, Factors & Metrics
 
### Testing Data
 
More information needed
 
### Factors
More information needed
 
### Metrics
 
More information needed
 
 
## Results 
 
More information needed

 
# Model Examination
 
More information needed
 
# Environmental Impact
 
Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).
 
- **Hardware Type:** More information needed
- **Hours used:** More information needed
- **Cloud Provider:** More information needed
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed
 
# Technical Specifications [optional]
 
## Model Architecture and Objective

The CLIP model devlopers note in their [model card](https://huggingface.co/openai/clip-vit-large-patch14) : 

> The base model uses a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.
 
## Compute Infrastructure
 
More information needed
 
### Hardware
 
 
More information needed
 
### Software
 
More information needed.
 
# Citation

 
**BibTeX:**
 
More information needed




**APA:**

More information needed
  
# Glossary [optional]
 
More information needed

# More Information [optional]
More information needed 

# Model Card Authors [optional]
 
CompVis  in collaboration with Ezi Ozoani and the Hugging Face team

# Model Card Contact
 
More information needed
 
# How to Get Started with the Model
 
Use the code below to get started with the model.
 
<details>
<summary> Click to expand </summary>

```python
from transformers import AutoProcessor, SafetyChecker
processor = AutoProcessor.from_pretrained("CompVis/stable-diffusion-safety-checker")
safety_checker = SafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
 ```
</details>