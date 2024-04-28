---
{}
---
This model is based on a custom Transformer model that can be installed with:

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

Now load the model and make predictions with:

```python
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

references = ["a bird chirps by the window", "this is a random sentence"]
candidates = ["a bird chirps by the window", "this looks like a random sentence"]

model.eval()
with torch.no_grad():
    inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    res = model(**inputs).logits.flatten().tolist()
print(res)
# [0.9990496635437012, 0.7930182218551636]
```

Take a look at this [repository](https://github.com/lucadiliello/bleurt-pytorch) for the definition of `BleurtConfig`, `BleurtForSequenceClassification` and `BleurtTokenizer` in PyTorch.