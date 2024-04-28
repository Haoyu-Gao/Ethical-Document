---
language:
- en
license: mit
library_name: transformers
tags:
- vision
- language
- fashion
- ecommerce
widget:
- src: https://cdn-images.farfetch-contents.com/19/76/05/56/19760556_44221665_1000.jpg
  candidate_labels: black shoe, red shoe, a cat
  example_title: Black Shoe
---

[![Youtube Video](https://img.shields.io/badge/youtube-video-red)](https://www.youtube.com/watch?v=uqRSc-KSA1Y) [![HuggingFace Model](https://img.shields.io/badge/HF%20Model-Weights-yellow)](https://huggingface.co/patrickjohncyh/fashion-clip) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z1hAxBnWjF76bEi9KQ6CMBBEmI_FVDrW?usp=sharing) [![Medium Blog Post](https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg)](https://towardsdatascience.com/teaching-clip-some-fashion-3005ac3fdcc3) [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/vinid/fashion-clip-app)

# Model Card: Fashion CLIP

Disclaimer: The model card adapts the model card from [here](https://huggingface.co/openai/clip-vit-base-patch32).

## Model Details

UPDATE (10/03/23): We have updated the model! We found that [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) checkpoint (thanks [Bin](https://www.linkedin.com/in/bin-duan-56205310/)!) worked better than original OpenAI CLIP on Fashion. We thus fine-tune a newer (and better!) version of FashionCLIP (henceforth FashionCLIP 2.0), while keeping the architecture the same. We postulate that the perofrmance gains afforded by `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` are due to the increased training data (5x OpenAI CLIP data). Our [thesis](https://www.nature.com/articles/s41598-022-23052-9), however, remains the same -- fine-tuning `laion/CLIP` on our fashion dataset improved zero-shot perofrmance across our benchmarks. See the below table comparing weighted macro F1 score across models.


| Model             | FMNIST        | KAGL          | DEEP          | 
| -------------     | ------------- | ------------- | ------------- |
| OpenAI CLIP       | 0.66          | 0.63          | 0.45          |
| FashionCLIP       | 0.74          | 0.67          | 0.48          |
| Laion CLIP        | 0.78          | 0.71          | 0.58          |
| FashionCLIP 2.0   | __0.83__          | __0.73__          | __0.62__          |

---

FashionCLIP is a CLIP-based model developed to produce general product representations for fashion concepts. Leveraging the pre-trained checkpoint (ViT-B/32) released by [OpenAI](https://github.com/openai/CLIP), we train FashionCLIP on a large, high-quality novel fashion dataset to study whether domain specific fine-tuning of CLIP-like models is sufficient to produce product representations that are zero-shot transferable to entirely new datasets and tasks. FashionCLIP was not developed for model deplyoment - to do so, researchers will first need to carefully study their capabilities in relation to the specific context they’re being deployed within.

### Model Date

March 2023

### Model Type

The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained, starting from a pre-trained checkpoint, to maximize the similarity of (image, text) pairs via a contrastive loss on a fashion dataset containing 800K products.


### Documents

- [FashionCLIP Github Repo](https://github.com/patrickjohncyh/fashion-clip)
- [FashionCLIP Paper](https://www.nature.com/articles/s41598-022-23052-9)


## Data

The model was trained on (image, text) pairs obtained from the Farfecth dataset[^1 Awaiting official release.], an English dataset comprising over 800K fashion products, with more than 3K brands across dozens of object types. The image used for encoding is the standard product image, which is a picture of the item over a white background, with no humans. The text used is a concatenation of the _highlight_ (e.g., “stripes”, “long sleeves”, “Armani”) and _short description_ (“80s styled t-shirt”)) available in the Farfetch dataset.



## Limitations, Bias and Fiarness

We acknowledge certain limitations of FashionCLIP and expect that it inherits certain limitations and biases present in the original CLIP model. We do not expect our fine-tuning to significantly augment these limitations: we acknowledge that the fashion data we use makes explicit assumptions about the notion of gender as in "blue shoes for a woman" that inevitably associate aspects of clothing with specific people.

Our investigations also suggest that the data used introduces certain limitations in FashionCLIP. From the textual modality, given that most captions derived from the Farfetch dataset are long, we observe that FashionCLIP may be more performant in longer queries than shorter ones. From the image modality, FashionCLIP is also biased towards standard product images (centered, white background).

Model selection, i.e. selecting an appropariate stopping critera during fine-tuning, remains an open challenge. We observed that using loss on an in-domain (i.e. same distribution as test) validation dataset is a poor selection critera when out-of-domain generalization (i.e. across different datasets) is desired, even when the dataset used is relatively diverse and large.


## Citation
```
@Article{Chia2022,
    title="Contrastive language and vision learning of general fashion concepts",
    author="Chia, Patrick John
            and Attanasio, Giuseppe
            and Bianchi, Federico
            and Terragni, Silvia
            and Magalh{\~a}es, Ana Rita
            and Goncalves, Diogo
            and Greco, Ciro
            and Tagliabue, Jacopo",
    journal="Scientific Reports",
    year="2022",
    month="Nov",
    day="08",
    volume="12",
    number="1",
    abstract="The steady rise of online shopping goes hand in hand with the development of increasingly complex ML and NLP models. While most use cases are cast as specialized supervised learning problems, we argue that practitioners would greatly benefit from general and transferable representations of products. In this work, we build on recent developments in contrastive learning to train FashionCLIP, a CLIP-like model adapted for the fashion industry. We demonstrate the effectiveness of the representations learned by FashionCLIP with extensive tests across a variety of tasks, datasets and generalization probes. We argue that adaptations of large pre-trained models such as CLIP offer new perspectives in terms of scalability and sustainability for certain types of players in the industry. Finally, we detail the costs and environmental impact of training, and release the model weights and code as open source contribution to the community.",
    issn="2045-2322",
    doi="10.1038/s41598-022-23052-9",
    url="https://doi.org/10.1038/s41598-022-23052-9"
}
```