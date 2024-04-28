---
language:
- ru
license: apache-2.0
tags:
- PyTorch
- Transformers
- bert
- exbert
pipeline_tag: fill-mask
thumbnail: https://github.com/sberbank-ai/model-zoo
---

# ruBert-base
The model architecture design, pretraining, and evaluation are documented in our preprint: [**A Family of Pretrained Transformer Language Models for Russian**](https://arxiv.org/abs/2309.10931).

The model is pretrained by the [SberDevices](https://sberdevices.ru/) team.  
* Task: `mask filling`
* Type: `encoder`
* Tokenizer: `BPE`
* Dict size: `120 138`
* Num Parameters: `178 M`	
* Training Data Volume `30 GB`

# Authors
+ NLP core team RnD [Telegram channel](https://t.me/nlpcoreteam):
  + Dmitry Zmitrovich
 
# Cite us
```
@misc{zmitrovich2023family,
      title={A Family of Pretrained Transformer Language Models for Russian}, 
      author={Dmitry Zmitrovich and Alexander Abramov and Andrey Kalmykov and Maria Tikhonova and Ekaterina Taktasheva and Danil Astafurov and Mark Baushenko and Artem Snegirev and Tatiana Shavrina and Sergey Markov and Vladislav Mikhailov and Alena Fenogenova},
      year={2023},
      eprint={2309.10931},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```