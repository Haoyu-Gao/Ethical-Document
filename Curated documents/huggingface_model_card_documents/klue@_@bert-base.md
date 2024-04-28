---
language: ko
license: cc-by-sa-4.0
tags:
- korean
- klue
mask_token: '[MASK]'
widget:
- text: 대한민국의 수도는 [MASK] 입니다.
---

# KLUE BERT base

## Table of Contents
- [Model Details](#model-details)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation Information](#citation-information)
- [Model Card Authors](#model-card-authors)

## Model Details

**Model Description:** KLUE BERT base is a pre-trained BERT Model on Korean Language. The developers of KLUE BERT base developed the model in the context of the development of the [Korean Language Understanding Evaluation (KLUE) Benchmark](https://arxiv.org/pdf/2105.09680.pdf).

- **Developed by:** See [GitHub Repo](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta) for model developers
- **Model Type:** Transformer-based language model
- **Language(s):** Korean
- **License:** cc-by-sa-4.0
- **Parent Model:** See the [BERT base uncased model](https://huggingface.co/bert-base-uncased) for more information about the BERT base model.
- **Resources for more information:**
  - [Research Paper](https://arxiv.org/abs/2105.09680)
  - [GitHub Repo](https://github.com/KLUE-benchmark/KLUE)

## How to Get Started With the Model

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
```

## Uses 

#### Direct Use

The model can be used for tasks including topic classification, semantic textual similarity, natural language inference, named entity recognition, and other tasks outlined in the [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE).

#### Misuse and Out-of-scope Use

The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Risks, Limitations and Biases

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). The model developers discuss several ethical considerations related to the model in the [paper](https://arxiv.org/pdf/2105.09680.pdf), including: 

- Bias issues with the publicly available data used in the pretraining corpora (and considerations related to filtering)
- PII in the data used in the pretraining corpora (and efforts to pseudonymize the data)

For ethical considerations related to the KLUE Benchmark, also see the [paper](https://arxiv.org/pdf/2105.09680.pdf).

## Training

#### Training Data

The authors use the following pretraining corpora for the model, described in the [associated paper](https://arxiv.org/pdf/2105.09680.pdf): 

> We gather the following five publicly available Korean corpora from diverse sources to cover a broad set of topics and many different styles. We combine these corpora to build the final pretraining corpus of size approximately 62GB. 
>
> - **MODU:** [Modu Corpus](https://corpus.korean.go.kr) is a collection of Korean corpora distributed by [National Institute of Korean Languages](https://corpus.korean.go.kr/). It includes both formal articles (news and books) and colloquial text (dialogues).
> - **CC-100-Kor:** [CC-100](https://data.statmt.org/cc-100/) is the large-scale multilingual web crawled corpora by using CC-Net ([Wenzek et al., 2020](https://www.aclweb.org/anthology/2020.lrec-1.494)). This is used for training XLM-R ([Conneau et al., 2020](https://aclanthology.org/2020.acl-main.747/)). We use the Korean portion from this corpora.
> - **NAMUWIKI:** NAMUWIKI is a Korean web-based encyclopedia, similar to Wikipedia, but known to be less formal. Specifically, we download [the dump](http://dump.thewiki.kr) created on March 2nd, 2020.
> - **NEWSCRAWL:** NEWSCRAWL consists of 12,800,000 news articles published from 2011 to 2020, collected from a news aggregation platform.
> - **PETITION:** Petition is a collection of public petitions posted to the Blue House asking for administrative actions on social issues. We use the articles in the [Blue House National Petition](https://www1.president.go.kr/petitions) published from [August 2017 to March 2019](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/korean_petitions.html).

The authors also describe ethical considerations related to the pretraining corpora in the [associated paper](https://arxiv.org/pdf/2105.09680.pdf).

#### Training Procedure

##### Preprocessing

The authors describe their preprocessing procedure in the [associated paper](https://arxiv.org/pdf/2105.09680.pdf):

> We filter noisy text and non-Korean text using the same methods from Section 2.3 (of the paper). Each document in the corpus is split into sentences using C++ implementation (v1.3.1.) of rule-based [Korean Sentence Splitter (KSS)](https://github.com/likejazz/korean-sentence-splitter). For CC-100-Kor and NEWSCRAWL, we keep sentences of length greater than equal to 200 characters, as a heuristics to keep well-formed sentences. We then remove sentences included in our benchmark task datasets, using BM25 as a sentence similarity metric ([reference](https://www.microsoft.com/en-us/research/publication/okapi-at-trec-3/)).

###### Tokenization

The authors describe their tokenization procedure in the [associated paper](https://arxiv.org/pdf/2105.09680.pdf):

> We design and use a new tokenization method, morpheme-based subword tokenization. When building a vocabulary, we pre-tokenize a raw text into morphemes using a morphological analyzer, and then we apply byte pair encoding (BPE) ([Senrich et al., 2016](https://aclanthology.org/P16-1162/)) to get the final vocabulary. For morpheme segmentation, we use [Mecab-ko](https://bitbucket.org/eunjeon/mecab-ko), MeCab ([Kudo, 2006](https://taku910.github.io/mecab/)) adapted for Korean, and for BPE segmentation, we use the wordpiece tokenizer from [Huggingface Tokenizers library](https://github.com/huggingface/tokenizers). We specify the vocabulary size to 32k. After building the vocabulary, we only use the BPE model during inference, which allows us to tokenize a word sequence by reflecting morphemes without a morphological analyzer. This improves both usability and speed.

The training configurations are further described in the [paper](https://arxiv.org/pdf/2105.09680.pdf).

## Evaluation

#### Testing Data, Factors and Metrics

The model was evaluated on the [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE). The tasks and metrics from the KLUE Benchmark that were used to evaluate this model are described briefly below. For more information about the KLUE Benchmark, see the [data card](https://huggingface.co/datasets/klue), [Github Repository](https://github.com/KLUE-benchmark/KLUE), and [associated paper](https://arxiv.org/pdf/2105.09680.pdf).

- **Task:** Topic Classification (TC) - Yonhap News Agency Topic Classification (YNAT), **Metrics:** Macro F1 score, defined as the mean of topic-wise F1 scores, giving the same importance to each topic.
  
- **Task:** Semantic Textual Similarity (STS), **Metrics:** Pearsons' correlation coefficient (Pearson’ r) and F1 score
  
- **Task:** Natural Language Inference (NLI), **Metrics:** Accuracy
  
- **Task:** Named Entity Recognition (NER), **Metrics:** Entity-level macro F1 (Entity F1) and character-level macro F1 (Char F1) scores

- **Task:** Relation Extraction (RE), **Metrics:** Micro F1 score on relation existing cases and area under the precision- recall curve (AUPRC) on all classes

- **Task:** Dependency Parsing (DP), **Metrics:** Unlabeled attachment score (UAS) and labeled attachment score (LAS)
   
- **Task:** Machine Reading Comprehension (MRC), **Metrics:** Exact match (EM) and character-level ROUGE-W (ROUGE), which can be viewed as longest common consecutive subsequence (LCCS)-based F1 score.
  
- **Task:** Dialogue State Tracking (DST), **Metrics:** Joint goal accuracy (JGA) and slot micro F1 score (Slot F1)

#### Results

| Task               | TC   |   STS      |       | NLI | NER        |         | RE  |      |   DP  |      |  MRC  |      |  DST   |        |
| :---:              |:---: | :---:      | :---: |:---:| :---:      | :---:   |:---:| :---:| :---: |:---: | :---: | :---:|  :---: |  :---: |
|  Metric            | F1   | Pearsons' r| F1    | ACC | Entity F1  | Char F1 |  F1 | AUPRC|  UAS  | LAS  |  EM   | ROUGE| JGA    |Slot F1 |
|                    | 85.73|   90.85    | 82.84 |81.63|   83.97    |   91.39 |66.44| 66.17| 89.96 |88.05 | 62.32 | 68.51| 46.64  | 91.61  |


## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). We present the hardware type based on the [associated paper](https://arxiv.org/pdf/2105.09680.pdf).

- **Hardware Type:** TPU v3-8
- **Hours used:** Unknown
- **Cloud Provider:** Unknown
- **Compute Region:** Unknown
- **Carbon Emitted:** Unknown

## Technical Specifications

See the [associated paper](https://arxiv.org/pdf/2105.09680.pdf) for details on the modeling architecture (BERT), objective, compute infrastructure, and training details.

## Citation Information

```bibtex
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
