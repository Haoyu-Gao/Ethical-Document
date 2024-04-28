---
language:
- ru
tags:
- PyTorch
- Transformers
thumbnail: https://github.com/sberbank-ai/ru-gpts
---

# rugpt3small\_based\_on\_gpt2
The model architecture design, pretraining, and evaluation are documented in our preprint: [**A Family of Pretrained Transformer Language Models for Russian**](https://arxiv.org/abs/2309.10931).

The model was pretrained with sequence length 1024 using transformers by the [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epochs. After that, the model was finetuned with the context size of 2048.

Total training time took around one week on 32 GPUs.

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