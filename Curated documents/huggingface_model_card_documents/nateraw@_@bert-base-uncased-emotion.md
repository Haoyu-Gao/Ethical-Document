---
language:
- en
license: apache-2.0
tags:
- text-classification
- emotion
- pytorch
datasets:
- emotion
metrics:
- accuracy
thumbnail: https://avatars3.githubusercontent.com/u/32437151?s=460&u=4ec59abc8d21d5feea3dab323d23a5860e6996a4&v=4
---

# bert-base-uncased-emotion

## Model description

`bert-base-uncased` finetuned on the emotion dataset using PyTorch Lightning. Sequence length 128, learning rate 2e-5, batch size 32, 2 GPUs, 4 epochs.

For more details, please see, [the emotion dataset on nlp viewer](https://huggingface.co/nlp/viewer/?dataset=emotion).


#### Limitations and bias

- Not the best model, but it works in a pinch I guess...
- Code not available as I just hacked this together.
- [Follow me on github](https://github.com/nateraw) to get notified when code is made available.

## Training data

Data came from HuggingFace's `datasets` package. The data can be viewed [on nlp viewer](https://huggingface.co/nlp/viewer/?dataset=emotion).


## Training procedure
...

## Eval results

val_acc - 0.931 (useless, as this should be precision/recall/f1)

The score was calculated using PyTorch Lightning metrics.
