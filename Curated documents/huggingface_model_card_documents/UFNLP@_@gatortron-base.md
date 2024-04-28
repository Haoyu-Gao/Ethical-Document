---
license: apache-2.0
---
<h2>GatorTron-Base overview </h2>

Developed by a joint effort between the University of Florida and NVIDIA, GatorTron-Base is a clinical language model of 345 million parameters, pre-trained using a BERT architecure implemented in the Megatron package (https://github.com/NVIDIA/Megatron-LM). 

GatorTron-Base is pre-trained using a dataset consisting of:

- 82B words of de-identified clinical notes from the University of Florida Health System,
- 6.1B words from PubMed CC0,
- 2.5B words from WikiText,
- 0.5B words of de-identified clinical notes from MIMIC-III

The Github for GatorTron is at : https://github.com/uf-hobi-informatics-lab/GatorTron

This model is converted to Hugginface from : https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/gatortron_og

<h2>Model variations</h2>

Model | Parameter 
--- | --- 
[gatortron-base (this model)](https://huggingface.co/UFNLP/gatortron-base)| 345 million 
[gatortronS](https://huggingface.co/UFNLP/gatortronS) | 345 million
[gatortron-medium](https://huggingface.co/UFNLP/gatortron-medium) | 3.9 billion 
gatortron-large | 8.9 billion

<h2>How to use</h2>

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig

tokenizer= AutoTokenizer.from_pretrained('UFNLP/gatortron-base')
config=AutoConfig.from_pretrained('UFNLP/gatortron-base')
mymodel=AutoModel.from_pretrained('UFNLP/gatortron-base')

encoded_input=tokenizer("Bone scan:  Negative for distant metastasis.", return_tensors="pt")
encoded_output = mymodel(**encoded_input)
print (encoded_output)

```

- An NLP pacakge using GatorTron for clinical concept extraction (Named Entity Recognition): https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER
- An NLP pacakge using GatorTron for Relation Extraction: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction
- An NLP pacakge using GatorTron for extraction of social determinants of health (SDoH) from clinical narratives: https://github.com/uf-hobi-informatics-lab/SDoH_SODA

<h2>De-identification</h2>

We applied a de-identification system to remove protected health information (PHI) from clinical text. We adopted the safe-harbor method to identify 18 PHI categories defined in the Health Insurance Portability and Accountability Act (HIPAA) and replaced them with dummy strings (e.g., replace people’s names into [\*\*NAME\*\*]). 

The de-identifiation system is described in:

Yang X, Lyu T, Li Q, Lee C-Y, Bian J, Hogan WR, Wu Y†. A study of deep learning methods for de-identification of clinical notes in cross-institute settings. BMC Med Inform Decis Mak. 2020 Dec 5;19(5):232. https://www.ncbi.nlm.nih.gov/pubmed/31801524.

<h2>Citation info</h2>

Yang X, Chen A, PourNejatian N, Shin HC, Smith KE, Parisien C, Compas C, Martin C, Costa AB, Flores MG, Zhang Y, Magoc T, Harle CA, Lipori G, Mitchell DA, Hogan WR, Shenkman EA, Bian J, Wu Y†. A large language model for electronic health records. Npj Digit Med. Nature Publishing Group; . 2022 Dec 26;5(1):1–9. https://www.nature.com/articles/s41746-022-00742-2

- BibTeX entry
```
@article{yang2022large,
  title={A large language model for electronic health records},
  author={Yang, Xi and Chen, Aokun and PourNejatian, Nima and Shin, Hoo Chang and Smith, Kaleb E and Parisien, Christopher and Compas, Colin and Martin, Cheryl and Costa, Anthony B and Flores, Mona G and Zhang, Ying and Magoc, Tanja and Harle, Christopher A and Lipori, Gloria and Mitchell, Duane A and Hogan, William R and Shenkman, Elizabeth A and Bian, Jiang and Wu, Yonghui },
  journal={npj Digital Medicine},
  volume={5},
  number={1},
  pages={194},
  year={2022},
  publisher={Nature Publishing Group UK London}
} 
```

<h2>Contact</h2>

- Yonghui Wu: yonghui.wu@ufl.edu
- Cheng Peng: c.peng@ufl.edu