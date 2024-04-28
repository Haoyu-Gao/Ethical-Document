---
language:
- bn
licenses:
- cc-by-nc-sa-4.0
---

# BanglaBERT

This repository contains the pretrained discriminator checkpoint of the model **BanglaBERT**. This is an [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) discriminator model pretrained with the Replaced Token Detection (RTD) objective. Finetuned models using this checkpoint achieve state-of-the-art results on many of the NLP tasks in bengali. 

For finetuning on different downstream tasks such as `Sentiment classification`, `Named Entity Recognition`, `Natural Language Inference` etc., refer to the scripts in the official GitHub [repository](https://github.com/csebuetnlp/banglabert).

**Note**: This model was pretrained using a specific normalization pipeline available [here](https://github.com/csebuetnlp/normalizer). All finetuning scripts in the official GitHub repository uses this normalization by default. If you need to adapt the pretrained model for a different task make sure the text units are normalized using this pipeline before tokenizing to get best results. A basic example is given below:

## Using this model as a discriminator in `transformers` (tested on 4.11.0.dev0)

```python
from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer
import torch

model = AutoModelForPreTraining.from_pretrained("csebuetnlp/banglabert")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

original_sentence = "আমি কৃতজ্ঞ কারণ আপনি আমার জন্য অনেক কিছু করেছেন।"
fake_sentence = "আমি হতাশ কারণ আপনি আমার জন্য অনেক কিছু করেছেন।"
fake_sentence = normalize(fake_sentence) # this normalization step is required before tokenizing the text

fake_tokens = tokenizer.tokenize(fake_sentence)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
discriminator_outputs = model(fake_inputs).logits
predictions = torch.round((torch.sign(discriminator_outputs) + 1) / 2)

[print("%7s" % token, end="") for token in fake_tokens]
print("\n" + "-" * 50)
[print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()[1:-1]]
print("\n" + "-" * 50)
```

## Benchmarks
 
* Zero-shot cross-lingual transfer-learning

|     Model          |   Params   |     SC (macro-F1)     |      NLI (accuracy)     |    NER  (micro-F1)   |   QA (EM/F1)   |   BangLUE score |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|[mBERT](https://huggingface.co/bert-base-multilingual-cased) | 180M  | 27.05 | 62.22 | 39.27 | 59.01/64.18 |  50.35 |
|[XLM-R (base)](https://huggingface.co/xlm-roberta-base) |  270M   | 42.03 | 72.18 | 45.37 | 55.03/61.83 |  55.29 |
|[XLM-R (large)](https://huggingface.co/xlm-roberta-large) | 550M  | 49.49 | 78.13 | 56.48 | 71.13/77.70 |  66.59 |
|[BanglishBERT](https://huggingface.co/csebuetnlp/banglishbert) | 110M | 48.39 | 75.26 | 55.56 | 72.87/78.63 | 66.14 |

* Supervised fine-tuning

|     Model          |   Params   |     SC (macro-F1)     |      NLI (accuracy)     |    NER  (micro-F1)   |   QA (EM/F1)   |   BangLUE score |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|[mBERT](https://huggingface.co/bert-base-multilingual-cased) | 180M  | 67.59 | 75.13 | 68.97 | 67.12/72.64 | 70.29 |
|[XLM-R (base)](https://huggingface.co/xlm-roberta-base) |  270M   | 69.54 | 78.46 | 73.32 | 68.09/74.27  | 72.82 |        
|[XLM-R (large)](https://huggingface.co/xlm-roberta-large) | 550M  | 70.97 | 82.40 | 78.39 | 73.15/79.06 | 76.79 |
|[sahajBERT](https://huggingface.co/neuropark/sahajBERT) | 18M | 71.12 | 76.92 | 70.94 | 65.48/70.69 | 71.03 |
|[BanglishBERT](https://huggingface.co/csebuetnlp/banglishbert) | 110M | 70.61 | 80.95 | 76.28 | 72.43/78.40 | 75.73 |
|[BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) | 110M | 72.89 | 82.80 | 77.78 | 72.63/79.34 | **77.09** |



The benchmarking datasets are as follows:
* **SC:** **[Sentiment Classification](https://aclanthology.org/2021.findings-emnlp.278)**
* **NER:** **[Named Entity Recognition](https://multiconer.github.io/competition)**
* **NLI:** **[Natural Language Inference](https://github.com/csebuetnlp/banglabert/#datasets)**
* **QA:** **[Question Answering](https://github.com/csebuetnlp/banglabert/#datasets)**

## Citation

If you use this model, please cite the following paper:
```
@inproceedings{bhattacharjee-etal-2022-banglabert,
    title = "{B}angla{BERT}: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in {B}angla",
    author = "Bhattacharjee, Abhik  and
      Hasan, Tahmid  and
      Ahmad, Wasi  and
      Mubasshir, Kazi Samin  and
      Islam, Md Saiful  and
      Iqbal, Anindya  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.98",
    pages = "1318--1327",
    abstract = "In this work, we introduce BanglaBERT, a BERT-based Natural Language Understanding (NLU) model pretrained in Bangla, a widely spoken yet low-resource language in the NLP literature. To pretrain BanglaBERT, we collect 27.5 GB of Bangla pretraining data (dubbed {`}Bangla2B+{'}) by crawling 110 popular Bangla sites. We introduce two downstream task datasets on natural language inference and question answering and benchmark on four diverse NLU tasks covering text classification, sequence labeling, and span prediction. In the process, we bring them under the first-ever Bangla Language Understanding Benchmark (BLUB). BanglaBERT achieves state-of-the-art results outperforming multilingual and monolingual models. We are making the models, datasets, and a leaderboard publicly available at \url{https://github.com/csebuetnlp/banglabert} to advance Bangla NLP.",
}
```

If you use the normalization module, please cite the following paper:
```
@inproceedings{hasan-etal-2020-low,
    title = "Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for {B}engali-{E}nglish Machine Translation",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Samin, Kazi  and
      Hasan, Masum  and
      Basak, Madhusudan  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.207",
    doi = "10.18653/v1/2020.emnlp-main.207",
    pages = "2612--2623",
    abstract = "Despite being the seventh most widely spoken language in the world, Bengali has received much less attention in machine translation literature due to being low in resources. Most publicly available parallel corpora for Bengali are not large enough; and have rather poor quality, mostly because of incorrect sentence alignments resulting from erroneous sentence segmentation, and also because of a high volume of noise present in them. In this work, we build a customized sentence segmenter for Bengali and propose two novel methods for parallel corpus creation on low-resource setups: aligner ensembling and batch filtering. With the segmenter and the two methods combined, we compile a high-quality Bengali-English parallel corpus comprising of 2.75 million sentence pairs, more than 2 million of which were not available before. Training on neural models, we achieve an improvement of more than 9 BLEU score over previous approaches to Bengali-English machine translation. We also evaluate on a new test set of 1000 pairs made with extensive quality control. We release the segmenter, parallel corpus, and the evaluation set, thus elevating Bengali from its low-resource status. To the best of our knowledge, this is the first ever large scale study on Bengali-English machine translation. We believe our study will pave the way for future research on Bengali-English machine translation as well as other low-resource languages. Our data and code are available at https://github.com/csebuetnlp/banglanmt.",
}
```


