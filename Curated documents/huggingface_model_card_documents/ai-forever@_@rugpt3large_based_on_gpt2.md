---
language:
- ru
tags:
- PyTorch
- Transformers
thumbnail: https://github.com/sberbank-ai/ru-gpts
---

# rugpt3large\_based\_on\_gpt2
The model architecture design, pretraining, and evaluation are documented in our preprint: [**A Family of Pretrained Transformer Language Models for Russian**](https://arxiv.org/abs/2309.10931).

The model was trained with sequence length 1024 using transformers lib by the [SberDevices](https://sberdevices.ru/) team on 80B tokens for 3 epochs. After that, the model was finetuned 1 epoch with sequence length 2048. 

Total training time was around 14 days on 128 GPUs for 1024 context and a few days on 16 GPUs for 2048 context.  
The final perplexity on the test set is `13.6`.

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