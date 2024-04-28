---
language:
- hr
- bs
- sr
- cnr
- hbs
license: apache-2.0
widget:
- text: Zovem se Marko i živim u Zagrebu. Studirao sam u Beogradu na Filozofskom fakultetu.
    Obožavam album Moanin.
---

# The [BERTić](https://huggingface.co/classla/bcms-bertic)&ast; [bert-ich] /bɜrtitʃ/ model fine-tuned for the task of named entity recognition in Bosnian, Croatian, Montenegrin and Serbian (BCMS)

&ast; The name should resemble the facts (1) that the model was trained in Zagreb, Croatia, where diminutives ending in -ić (as in fotić, smajlić, hengić etc.) are very popular, and (2) that most surnames in the countries where these languages are spoken end in -ić (with diminutive etymology as well).

This is a fine-tuned version of the [BERTić](https://huggingface.co/classla/bcms-bertic) model for the task of named entity recognition (PER, LOC, ORG, MISC). The fine-tuning was performed on the following datasets:

- the [hr500k](http://hdl.handle.net/11356/1183) dataset, 500 thousand tokens in size, standard Croatian
- the [SETimes.SR](http://hdl.handle.net/11356/1200) dataset, 87 thousand tokens in size, standard Serbian
- the [ReLDI-hr](http://hdl.handle.net/11356/1241) dataset, 89 thousand tokens in size, Internet (Twitter) Croatian
- the [ReLDI-sr](http://hdl.handle.net/11356/1240) dataset, 92 thousand tokens in size, Internet (Twitter) Serbian

The data was augmented with missing diacritics and standard data was additionally over-represented. The F1 obtained on dev data (train and test was merged into train) is 91.38. For a more detailed per-dataset evaluation of the BERTić model on the NER task have a look at the [main model page](https://huggingface.co/classla/bcms-bertic).

If you use this fine-tuned model, please cite the following paper:

```
@inproceedings{ljubesic-lauc-2021-bertic,
    title = "{BERT}i{\'c} - The Transformer Language Model for {B}osnian, {C}roatian, {M}ontenegrin and {S}erbian",
    author = "Ljube{\v{s}}i{\'c}, Nikola  and Lauc, Davor",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.5",
    pages = "37--42",
}
```

When running the model in `simpletransformers`, the order of labels has to be set as well.

```
from simpletransformers.ner import NERModel, NERArgs
model_args = NERArgs()
model_args.labels_list = ['B-LOC','B-MISC','B-ORG','B-PER','I-LOC','I-MISC','I-ORG','I-PER','O']
model = NERModel('electra', 'classla/bcms-bertic-ner', args=model_args)
```