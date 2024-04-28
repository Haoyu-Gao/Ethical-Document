---
language: cs
license: cc-by-4.0
---

# Small-E-Czech

Small-E-Czech is an [Electra](https://arxiv.org/abs/2003.10555)-small model pretrained on a Czech web corpus created at [Seznam.cz](https://www.seznam.cz/) and introduced in an [IAAI 2022 paper](https://arxiv.org/abs/2112.01810). Like other pretrained models, it should be finetuned on a downstream task of interest before use. At Seznam.cz, it has helped improve [web search ranking](https://blog.seznam.cz/2021/02/vyhledavani-pomoci-vyznamovych-vektoru/), query typo correction or clickbait titles detection. We release it under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) (i.e. allowing commercial use). To raise an issue, please visit our [github](https://github.com/seznam/small-e-czech).

### How to use the discriminator in transformers
```python
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("Seznam/small-e-czech")
tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

sentence = "Za hory, za doly, mé zlaté parohy"
fake_sentence = "Za hory, za doly, kočka zlaté parohy"

fake_sentence_tokens = ["[CLS]"] + tokenizer.tokenize(fake_sentence) + ["[SEP]"]
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
outputs = discriminator(fake_inputs)
predictions = torch.nn.Sigmoid()(outputs[0]).cpu().detach().numpy()

for token in fake_sentence_tokens:
    print("{:>7s}".format(token), end="")
print()

for prediction in predictions.squeeze():
    print("{:7.1f}".format(prediction), end="")
print()
```

In the output we can see the probabilities of particular tokens not belonging in the sentence (i.e. having been faked by the generator) according to the discriminator:

```
  [CLS]     za   hory      ,     za    dol    ##y      ,  kočka  zlaté   paro   ##hy  [SEP]
    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.8    0.3    0.2    0.1    0.0
```

### Finetuning

For instructions on how to finetune the model on a new task, see the official HuggingFace transformers [tutorial](https://huggingface.co/transformers/training.html).