---
language: en
license: apache-2.0
tags:
- bert
- exbert
- linkbert
- biolinkbert
- feature-extraction
- fill-mask
- question-answering
- text-classification
- token-classification
datasets:
- pubmed
widget:
- text: Sunitinib is a tyrosine kinase inhibitor
---

## BioLinkBERT-base

BioLinkBERT-base model pretrained on [PubMed](https://pubmed.ncbi.nlm.nih.gov/) abstracts along with citation link information. It is introduced in the paper [LinkBERT: Pretraining Language Models with Document Links (ACL 2022)](https://arxiv.org/abs/2203.15827). The code and data are available in [this repository](https://github.com/michiyasunaga/LinkBERT).

This model achieves state-of-the-art performance on several biomedical NLP benchmarks such as [BLURB](https://microsoft.github.io/BLURB/) and [MedQA-USMLE](https://github.com/jind11/MedQA).


## Model description

LinkBERT is a transformer encoder (BERT-like) model pretrained on a large corpus of documents. It is an improvement of BERT that newly captures **document links** such as hyperlinks and citation links to include knowledge that spans across multiple documents. Specifically, it was pretrained by feeding linked documents into the same language model context, besides a single document.

LinkBERT can be used as a drop-in replacement for BERT. It achieves better performance for general language understanding tasks (e.g. text classification), and is also particularly effective for **knowledge-intensive** tasks (e.g. question answering) and **cross-document** tasks (e.g. reading comprehension, document retrieval).


## Intended uses & limitations

The model can be used by fine-tuning on a downstream task, such as question answering, sequence classification, and token classification.
You can also use the raw model for feature extraction (i.e. obtaining embeddings for input text).


### How to use

To use the model to get the features of a given text in PyTorch:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')
inputs = tokenizer("Sunitinib is a tyrosine kinase inhibitor", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

For fine-tuning, you can use [this repository](https://github.com/michiyasunaga/LinkBERT) or follow any other BERT fine-tuning codebases.


## Evaluation results

When fine-tuned on downstream tasks, LinkBERT achieves the following results.

**Biomedical benchmarks ([BLURB](https://microsoft.github.io/BLURB/), [MedQA](https://github.com/jind11/MedQA), [MMLU](https://github.com/hendrycks/test), etc.):** BioLinkBERT attains new state-of-the-art.

|                         | BLURB score | PubMedQA | BioASQ   | MedQA-USMLE |
| ----------------------  | --------    | -------- | -------  | --------    |
| PubmedBERT-base         | 81.10       | 55.8     | 87.5     | 38.1        |
| **BioLinkBERT-base**    | **83.39**   | **70.2** | **91.4** | **40.0** |
| **BioLinkBERT-large**   | **84.30**   | **72.2** | **94.8** | **44.6** |

|                         | MMLU-professional medicine     |
| ----------------------  | --------  |
| GPT-3 (175 params)      | 38.7      |
| UnifiedQA (11B params)  | 43.2      |
| **BioLinkBERT-large (340M params)** | **50.7**  |


## Citation

If you find LinkBERT useful in your project, please cite the following:

```bibtex
@InProceedings{yasunaga2022linkbert,
  author =  {Michihiro Yasunaga and Jure Leskovec and Percy Liang},
  title =   {LinkBERT: Pretraining Language Models with Document Links},
  year =    {2022},  
  booktitle = {Association for Computational Linguistics (ACL)},  
}
```
