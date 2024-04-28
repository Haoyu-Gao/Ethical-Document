---
language:
- es
library_name: pysentimiento
tags:
- emotion-analysis
- twitter
---

# Emotion Analysis in Spanish
## robertuito-emotion-analysis

Repository: [https://github.com/pysentimiento/pysentimiento/](https://github.com/finiteautomata/pysentimiento/)



Model trained with TASS 2020 Task 2 corpus for Emotion detection in Spanish. Base model is [RoBERTuito](https://github.com/pysentimiento/robertuito), a RoBERTa model trained in Spanish tweets.

Contains the six Ekman emotions plus a neutral class:

- anger
- disgust
- fear
- joy
- sadness
- surprise


## Results

Results for the four tasks evaluated in `pysentimiento`. Results are expressed as Macro F1 scores


| model         | emotion       | hate_speech   | irony         | sentiment     |
|:--------------|:--------------|:--------------|:--------------|:--------------|
| robertuito    | 0.560 ± 0.010 | 0.759 ± 0.007 | 0.739 ± 0.005 | 0.705 ± 0.003 |
| roberta       | 0.527 ± 0.015 | 0.741 ± 0.012 | 0.721 ± 0.008 | 0.670 ± 0.006 |
| bertin        | 0.524 ± 0.007 | 0.738 ± 0.007 | 0.713 ± 0.012 | 0.666 ± 0.005 |
| beto_uncased  | 0.532 ± 0.012 | 0.727 ± 0.016 | 0.701 ± 0.007 | 0.651 ± 0.006 |
| beto_cased    | 0.516 ± 0.012 | 0.724 ± 0.012 | 0.705 ± 0.009 | 0.662 ± 0.005 |
| mbert_uncased | 0.493 ± 0.010 | 0.718 ± 0.011 | 0.681 ± 0.010 | 0.617 ± 0.003 |
| biGRU         | 0.264 ± 0.007 | 0.592 ± 0.018 | 0.631 ± 0.011 | 0.585 ± 0.011 |


Note that for Hate Speech, these are the results for Semeval 2019, Task 5 Subtask B (HS+TR+AG detection)

## Citation

If you use this model in your research, please cite pysentimiento, RoBERTuito and EmoEvent papers:

```
@misc{perez2021pysentimiento,
      title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks},
      author={Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
      year={2021},
      eprint={2106.09462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{del2020emoevent,
  title={EmoEvent: A multilingual emotion corpus based on different events},
  author={del Arco, Flor Miriam Plaza and Strapparava, Carlo and Lopez, L Alfonso Urena and Mart{\'\i}n-Valdivia, M Teresa},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={1492--1498},
  year={2020}
}

@inproceedings{perez-etal-2022-robertuito,
    title = "{R}o{BERT}uito: a pre-trained language model for social media text in {S}panish",
    author = "P{\'e}rez, Juan Manuel  and
      Furman, Dami{\'a}n Ariel  and
      Alonso Alemany, Laura  and
      Luque, Franco M.",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.785",
    pages = "7235--7243",
    abstract = "Since BERT appeared, Transformer language models and transfer learning have become state-of-the-art for natural language processing tasks. Recently, some works geared towards pre-training specially-crafted models for particular domains, such as scientific papers, medical documents, user-generated texts, among others. These domain-specific models have been shown to improve performance significantly in most tasks; however, for languages other than English, such models are not widely available. In this work, we present RoBERTuito, a pre-trained language model for user-generated text in Spanish, trained on over 500 million tweets. Experiments on a benchmark of tasks involving user-generated text showed that RoBERTuito outperformed other pre-trained language models in Spanish. In addition to this, our model has some cross-lingual abilities, achieving top results for English-Spanish tasks of the Linguistic Code-Switching Evaluation benchmark (LinCE) and also competitive performance against monolingual models in English Twitter tasks. To facilitate further research, we make RoBERTuito publicly available at the HuggingFace model hub together with the dataset used to pre-train it.",
}
```