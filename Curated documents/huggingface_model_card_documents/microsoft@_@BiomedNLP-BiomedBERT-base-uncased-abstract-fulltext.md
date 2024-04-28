---
language: en
license: mit
tags:
- exbert
widget:
- text: '[MASK] is a tumor suppressor gene.'
---

## MSR BiomedBERT (abstracts + full text)

<div style="border: 2px solid orange; border-radius:10px; padding:0px 10px; width: fit-content;">

* This model was previously named **"PubMedBERT (abstracts + full text)"**.
* You can either adopt the new model name "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" or update your `transformers` library to version 4.22+ if you need to refer to the old name.

</div>

Pretraining large neural language models, such as BERT, has led to impressive gains on many natural language processing (NLP) tasks. However, most pretraining efforts focus on general domain corpora, such as newswire and Web. A prevailing assumption is that even domain-specific pretraining can benefit by starting from general-domain language models. [Recent work](https://arxiv.org/abs/2007.15779) shows that for domains with abundant unlabeled text, such as biomedicine, pretraining language models from scratch results in substantial gains over continual pretraining of general-domain language models.

BiomedBERT is pretrained from scratch using _abstracts_ from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) and _full-text_ articles from [PubMedCentral](https://www.ncbi.nlm.nih.gov/pmc/). This model achieves state-of-the-art performance on many biomedical NLP tasks, and currently holds the top score on the [Biomedical Language Understanding and Reasoning Benchmark](https://aka.ms/BLURB).

## Citation

If you find BiomedBERT useful in your research, please cite the following paper:

```latex
@misc{pubmedbert,
  author = {Yu Gu and Robert Tinn and Hao Cheng and Michael Lucas and Naoto Usuyama and Xiaodong Liu and Tristan Naumann and Jianfeng Gao and Hoifung Poon},
  title = {Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing},
  year = {2020},
  eprint = {arXiv:2007.15779},
}
```

<a href="https://huggingface.co/exbert/?model=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext&modelKind=bidirectional&sentence=Gefitinib%20is%20an%20EGFR%20tyrosine%20kinase%20inhibitor,%20which%20is%20often%20used%20for%20breast%20cancer%20and%20NSCLC%20treatment.&layer=3&heads=..0,1,2,3,4,5,6,7,8,9,10,11&threshold=0.7&tokenInd=17&tokenSide=right&maskInds=..&hideClsSep=true">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
