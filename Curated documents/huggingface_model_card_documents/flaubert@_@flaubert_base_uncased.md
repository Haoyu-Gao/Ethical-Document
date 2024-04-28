---
language: fr
license: mit
tags:
- bert
- language-model
- flaubert
- flue
- french
- flaubert-base
- uncased
datasets:
- flaubert
metrics:
- flue
---

# FlauBERT: Unsupervised Language Model Pre-training for French

**FlauBERT** is a French BERT trained on a very large and heterogeneous French corpus. Models of different sizes are trained using the new CNRS (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/eng/jean-zay/ ) supercomputer.
 
Along with FlauBERT comes [**FLUE**](https://github.com/getalp/Flaubert/tree/master/flue): an evaluation setup for French NLP systems similar to the popular GLUE benchmark. The goal is to enable further reproducible experiments in the future and to share models and progress on the French language.For more details please refer to the [official website](https://github.com/getalp/Flaubert).

## FlauBERT models

| Model name | Number of layers | Attention Heads | Embedding Dimension | Total Parameters |
| :------:       |   :---: | :---: | :---: | :---: |
| `flaubert-small-cased` | 6    | 8    | 512   | 54 M |
| `flaubert-base-uncased`  | 12  | 12  | 768  | 137 M |
| `flaubert-base-cased`   | 12   | 12      | 768   | 138 M |
| `flaubert-large-cased`  | 24   | 16     | 1024 | 373 M |

**Note:** `flaubert-small-cased` is partially trained so performance is not guaranteed. Consider using it for debugging purpose only.

## Using FlauBERT with Hugging Face's Transformers

```python
import torch
from transformers import FlaubertModel, FlaubertTokenizer

# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased', 
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_base_cased' 

# Load pretrained model and tokenizer
flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
# do_lowercase=False if using cased models, True if using uncased ones

sentence = "Le chat mange une pomme."
token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])

last_layer = flaubert(token_ids)[0]
print(last_layer.shape)
# torch.Size([1, 8, 768])  -> (batch size x number of tokens x embedding dimension)

# The BERT [CLS] token correspond to the first hidden state of the last layer
cls_embedding = last_layer[:, 0, :]
```

**Notes:** if your `transformers` version is <=2.10.0, `modelname` should take one
of the following values:

```
['flaubert-small-cased', 'flaubert-base-uncased', 'flaubert-base-cased', 'flaubert-large-cased']
```


## References

If you use FlauBERT or the FLUE Benchmark for your scientific publication, or if you find the resources in this repository useful, please cite one of the following papers:

[LREC paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.302.pdf)
```
@InProceedings{le2020flaubert,
  author    = {Le, Hang  and  Vial, Lo\"{i}c  and  Frej, Jibril  and  Segonne, Vincent  and  Coavoux, Maximin  and  Lecouteux, Benjamin  and  Allauzen, Alexandre  and  Crabb\'{e}, Beno\^{i}t  and  Besacier, Laurent  and  Schwab, Didier},
  title     = {FlauBERT: Unsupervised Language Model Pre-training for French},
  booktitle = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month     = {May},
  year      = {2020},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {2479--2490},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.302}
}
```

[TALN paper](https://hal.archives-ouvertes.fr/hal-02784776/)
```
@inproceedings{le2020flaubert,
  title         = {FlauBERT: des mod{\`e}les de langue contextualis{\'e}s pr{\'e}-entra{\^\i}n{\'e}s pour le fran{\c{c}}ais},
  author        = {Le, Hang and Vial, Lo{\"\i}c and Frej, Jibril and Segonne, Vincent and Coavoux, Maximin and Lecouteux, Benjamin and Allauzen, Alexandre and Crabb{\'e}, Beno{\^\i}t and Besacier, Laurent and Schwab, Didier},
  booktitle     = {Actes de la 6e conf{\'e}rence conjointe Journ{\'e}es d'{\'E}tudes sur la Parole (JEP, 31e {\'e}dition), Traitement Automatique des Langues Naturelles (TALN, 27e {\'e}dition), Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (R{\'E}CITAL, 22e {\'e}dition). Volume 2: Traitement Automatique des Langues Naturelles},
  pages         = {268--278},
  year          = {2020},
  organization  = {ATALA}
}
```