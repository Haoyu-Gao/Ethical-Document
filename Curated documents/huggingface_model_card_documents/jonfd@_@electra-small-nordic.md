---
language:
- is
- false
- sv
- da
license: cc-by-4.0
datasets:
- igc
- ic3
- jonfd/ICC
- mc4
---

# Nordic ELECTRA-Small
This model was pretrained on the following corpora:
* The [Icelandic Gigaword Corpus](http://igc.arnastofnun.is/) (IGC)
* The Icelandic Common Crawl Corpus (IC3)
* The [Icelandic Crawled Corpus](https://huggingface.co/datasets/jonfd/ICC) (ICC)
* The [Multilingual Colossal Clean Crawled Corpus](https://huggingface.co/datasets/mc4) (mC4) - Icelandic, Norwegian, Swedish and Danish text obtained from .is, .no, .se and .dk domains, respectively

The total size of the corpus after document-level deduplication and filtering was 14.82B tokens, split equally between the four languages. The model was trained using a WordPiece tokenizer with a vocabulary size of 96,105 for one million steps with a batch size of 256, and otherwise with default settings.

# Acknowledgments
This research was supported with Cloud TPUs from Google's TPU Research Cloud (TRC).

This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by [Almannarómur](https://almannaromur.is/), is funded by the Icelandic Ministry of Education, Science and Culture.