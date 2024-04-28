---
language:
- es
tags:
- sentiment-analysis
---

# Sentiment Analysis in Spanish
## beto-sentiment-analysis

**NOTE: this model will be removed soon -- use [pysentimiento/robertuito-sentiment-analysis](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis) instead**

Repository: [https://github.com/finiteautomata/pysentimiento/](https://github.com/pysentimiento/pysentimiento/)


Model trained with TASS 2020 corpus (around ~5k tweets) of several dialects of Spanish. Base model is [BETO](https://github.com/dccuchile/beto), a BERT model trained in Spanish.

Uses `POS`, `NEG`, `NEU` labels.

## License

`pysentimiento` is an open-source library for non-commercial use and scientific research purposes only. Please be aware that models are trained with third-party datasets and are subject to their respective licenses. 

1. [TASS Dataset license](http://tass.sepln.org/tass_data/download.php)
2. [SEMEval 2017 Dataset license]()

## Citation

If you use this model in your work, please cite the following papers:

```
@misc{perez2021pysentimiento,
      title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks},
      author={Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
      year={2021},
      eprint={2106.09462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{canete2020spanish,
  title={Spanish pre-trained bert model and evaluation data},
  author={Ca{\~n}ete, Jos{\'e} and Chaperon, Gabriel and Fuentes, Rodrigo and Ho, Jou-Hui and Kang, Hojin and P{\'e}rez, Jorge},
  journal={Pml4dc at iclr},
  volume={2020},
  number={2020},
  pages={1--10},
  year={2020}
}
```

Enjoy! 🤗
