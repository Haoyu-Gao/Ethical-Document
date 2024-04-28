---
language: multilingual
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
---

# sentence-transformers/clip-ViT-B-32-multilingual-v1

This is a multi-lingual version of the OpenAI CLIP-ViT-B32 model. You can map text (in 50+ languages) and images to a common dense vector space such that images and the matching texts are close. This model can be used for **image search** (users search through a large collection of images) and for **multi-lingual zero-shot image classification** (image labels are defined as text).


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


# Now we load and encode the images
def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

# We load 3 images. You can either pass URLs or
# a path on your disc
img_paths = [
    # Dog image
    "https://unsplash.com/photos/QtxgNsmJQSs/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjM1ODQ0MjY3&w=640",

    # Cat image
    "https://unsplash.com/photos/9UUoGaaHtNE/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8Mnx8Y2F0fHwwfHx8fDE2MzU4NDI1ODQ&w=640",

    # Beach image
    "https://unsplash.com/photos/Siuwr3uCir0/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8NHx8YmVhY2h8fDB8fHx8MTYzNTg0MjYzMg&w=640"
]

images = [load_image(img) for img in img_paths]

# Map images to the vector space
img_embeddings = img_model.encode(images)

# Now we encode our text:
texts = [
    "A dog in the snow",
    "Eine Katze",  # German: A cat
    "Una playa con palmeras."  # Spanish: a beach with palm trees
]

text_embeddings = text_model.encode(texts)

# Compute cosine similarities:
cos_sim = util.cos_sim(text_embeddings, img_embeddings)

for text, scores in zip(texts, cos_sim):
    max_img_idx = torch.argmax(scores)
    print("Text:", text)
    print("Score:", scores[max_img_idx] )
    print("Path:", img_paths[max_img_idx], "\n")

```

## Multilingual Image Search - Demo
For a demo of multilingual image search, have a look at: [Image_Search-multilingual.ipynb](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/image-search/Image_Search-multilingual.ipynb) ( [Colab version](https://colab.research.google.com/drive/1N6woBKL4dzYsHboDNqtv-8gjZglKOZcn?usp=sharing) )

For more details on image search and zero-shot image classification, have a look at the documentation on [SBERT.net](https://www.sbert.net/examples/applications/image-search/README.html).


## Training
This model has been created using [Multilingual Knowledge Distillation](https://arxiv.org/abs/2004.09813). As teacher model, we used the original `clip-ViT-B-32` and then trained a [multilingual DistilBERT](https://huggingface.co/distilbert-base-multilingual-cased) model as student model. Using parallel data, the multilingual student model learns to align the teachers vector space across many languages. As a result, you get an text embedding model that works for 50+ languages.

The image encoder from CLIP is unchanged, i.e. you can use the original CLIP image encoder to encode images.

Have a look at the [SBERT.net - Multilingual-Models documentation](https://www.sbert.net/examples/training/multilingual/README.html) on more details and for **training code**.

We used the following 50+ languages to align the vector spaces: ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw.

The original multilingual DistilBERT supports 100+ lanugages. The model also work for these languages, but might not yield the best results.

## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Dense({'in_features': 768, 'out_features': 512, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
)
```

## Citing & Authors

This model was trained by [sentence-transformers](https://www.sbert.net/). 
        
If you find this model helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):
```bibtex 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```