---
language:
- en
tags:
- sentiment-analysis
---
# Sentiment Analysis in English
## bertweet-sentiment-analysis

Repository: [https://github.com/finiteautomata/pysentimiento/](https://github.com/finiteautomata/pysentimiento/)


Model trained with SemEval 2017 corpus (around ~40k tweets). Base model is [BERTweet](https://github.com/VinAIResearch/BERTweet), a RoBERTa model trained on English tweets.

Uses `POS`, `NEG`, `NEU` labels.

## License

`pysentimiento` is an open-source library for non-commercial use and scientific research purposes only. Please be aware that models are trained with third-party datasets and are subject to their respective licenses. 

1. [TASS Dataset license](http://tass.sepln.org/tass_data/download.php)
2. [SEMEval 2017 Dataset license]()

## Citation

If you use `pysentimiento` in your work, please cite [this paper](https://arxiv.org/abs/2106.09462)

```
@misc{perez2021pysentimiento,
      title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks},
      author={Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
      year={2021},
      eprint={2106.09462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Enjoy! 🤗
