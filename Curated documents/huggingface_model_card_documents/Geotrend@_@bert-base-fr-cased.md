---
language: fr
license: apache-2.0
datasets: wikipedia
widget:
- text: Paris est la [MASK] de la France.
- text: Paris est la capitale de la [MASK].
- text: L'élection américaine a eu [MASK] en novembre 2020.
---

# bert-base-fr-cased

We are sharing smaller versions of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) that handle a custom number of languages.

Unlike [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased), our versions give exactly the same representations produced by the original model which preserves the original accuracy.

For more information please visit our paper: [Load What You Need: Smaller Versions of Multilingual BERT](https://www.aclweb.org/anthology/2020.sustainlp-1.16.pdf).

## How to use

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Geotrend/bert-base-fr-cased")
model = AutoModel.from_pretrained("Geotrend/bert-base-fr-cased")

```

To generate other smaller versions of multilingual transformers please visit [our Github repo](https://github.com/Geotrend-research/smaller-transformers).

### How to cite

```bibtex
@inproceedings{smallermbert,
  title={Load What You Need: Smaller Versions of Mutlilingual BERT},
  author={Abdaoui, Amine and Pradel, Camille and Sigel, Grégoire},
  booktitle={SustaiNLP / EMNLP},
  year={2020}
}
```

## Contact 

Please contact amine@geotrend.fr for any question, feedback or request.
