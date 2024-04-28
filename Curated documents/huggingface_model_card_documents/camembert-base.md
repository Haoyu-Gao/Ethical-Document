---
language: fr
license: mit
datasets:
- oscar
---

# CamemBERT: a Tasty French Language Model

## Table of Contents
- [Model Details](#model-details)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation Information](#citation-information)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)


## Model Details
- **Model Description:**
CamemBERT is a state-of-the-art language model for French based on the RoBERTa model.
It is now available on Hugging Face in 6 different versions with varying number of parameters, amount of pretraining data and pretraining data source domains.
- **Developed by:**  Louis Martin\*, Benjamin Muller\*, Pedro Javier Ortiz Suárez\*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
- **Model Type:** Fill-Mask
- **Language(s):** French
- **License:** MIT
- **Parent Model:** See the [RoBERTa base model](https://huggingface.co/roberta-base) for more information about the RoBERTa   base model.
- **Resources for more information:**
	- [Research Paper](https://arxiv.org/abs/1911.03894)
  - [Camembert Website](https://camembert-model.fr/)
  
  
## Uses

#### Direct Use

This model can be used for Fill-Mask tasks.


## Risks, Limitations and Biases
**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).

This model was pretrained on a subcorpus of OSCAR multilingual corpus. Some of the limitations and risks associated with the OSCAR dataset, which are further detailed in the [OSCAR dataset card](https://huggingface.co/datasets/oscar), include the following: 

> The quality of some OSCAR sub-corpora might be lower than expected, specifically for the lowest-resource languages.

> Constructed from Common Crawl, Personal and sensitive information might be present.



## Training


#### Training Data
OSCAR or Open Super-large Crawled Aggregated coRpus is a multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.


#### Training Procedure

| Model                          | #params                        | Arch. | Training data                     |
|--------------------------------|--------------------------------|-------|-----------------------------------|
| `camembert-base` | 110M   | Base  | OSCAR (138 GB of text)            |
| `camembert/camembert-large`              | 335M    | Large | CCNet (135 GB of text)            |
| `camembert/camembert-base-ccnet`         | 110M    | Base  | CCNet (135 GB of text)            |
| `camembert/camembert-base-wikipedia-4gb` | 110M    | Base  | Wikipedia (4 GB of text)          |
| `camembert/camembert-base-oscar-4gb`     | 110M    | Base  | Subsample of OSCAR (4 GB of text) |
| `camembert/camembert-base-ccnet-4gb`     | 110M    | Base  | Subsample of CCNet (4 GB of text) |

## Evaluation


The model developers evaluated CamemBERT using four different downstream tasks for French: part-of-speech (POS) tagging, dependency parsing, named entity recognition (NER) and natural language inference (NLI).



## Citation Information

```bibtex
@inproceedings{martin2020camembert,
  title={CamemBERT: a Tasty French Language Model},
  author={Martin, Louis and Muller, Benjamin and Su{\'a}rez, Pedro Javier Ortiz and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^\i}t},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

## How to Get Started With the Model

##### Load CamemBERT and its sub-word tokenizer :
```python
from transformers import CamembertModel, CamembertTokenizer

# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
camembert = CamembertModel.from_pretrained("camembert-base")

camembert.eval()  # disable dropout (or leave in train mode to finetune)

```

##### Filling masks using pipeline 
```python
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert-base", tokenizer="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
# results
#[{'sequence': '<s> Le camembert est délicieux :)</s>', 'score': 0.4909103214740753, 'token': 7200},
# {'sequence': '<s> Le camembert est excellent :)</s>', 'score': 0.10556930303573608, 'token': 2183}, 
# {'sequence': '<s> Le camembert est succulent :)</s>', 'score': 0.03453315049409866, 'token': 26202}, 
# {'sequence': '<s> Le camembert est meilleur :)</s>', 'score': 0.03303130343556404, 'token': 528}, 
# {'sequence': '<s> Le camembert est parfait :)</s>', 'score': 0.030076518654823303, 'token': 1654}]

```

##### Extract contextual embedding features from Camembert output 
```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁ca', 'member', 't', '▁!'] 

# 1-hot encode and add special starting and end tokens 
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 121, 11, 660, 16, 730, 25543, 110, 83, 6] 
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = camembert(encoded_sentence)
# embeddings.detach()
# embeddings.size torch.Size([1, 10, 768])
# tensor([[[-0.0254,  0.0235,  0.1027,  ..., -0.1459, -0.0205, -0.0116],
#         [ 0.0606, -0.1811, -0.0418,  ..., -0.1815,  0.0880, -0.0766],
#         [-0.1561, -0.1127,  0.2687,  ..., -0.0648,  0.0249,  0.0446],
#         ...,
```

##### Extract contextual embedding features from all Camembert layers
```python
from transformers import CamembertConfig
# (Need to reload the model with new config)
config = CamembertConfig.from_pretrained("camembert-base", output_hidden_states=True)
camembert = CamembertModel.from_pretrained("camembert-base", config=config)

embeddings, _, all_layer_embeddings = camembert(encoded_sentence)
#  all_layer_embeddings list of len(all_layer_embeddings) == 13 (input embedding layer + 12 self attention layers)
all_layer_embeddings[5]
# layer 5 contextual embedding : size torch.Size([1, 10, 768])
#tensor([[[-0.0032,  0.0075,  0.0040,  ..., -0.0025, -0.0178, -0.0210],
#         [-0.0996, -0.1474,  0.1057,  ..., -0.0278,  0.1690, -0.2982],
#         [ 0.0557, -0.0588,  0.0547,  ..., -0.0726, -0.0867,  0.0699],
#         ...,
```

