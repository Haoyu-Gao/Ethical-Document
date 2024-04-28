---
language:
- ru
- en
tags:
- feature-extraction
- embeddings
- sentence-similarity
---
# LaBSE for English and Russian
This is a truncated version of [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE), which is, in turn, a port of [LaBSE](https://tfhub.dev/google/LaBSE/1) by Google.

The current model has only English and Russian tokens left in the vocabulary.
Thus, the vocabulary is 10% of the original, and number of parameters in the whole model is 27% of the original, without any loss in the quality of English and Russian embeddings.
 
To get the sentence embeddings, you can  use the following code:
```python
import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
sentences = ["Hello World", "Привет Мир"]
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
embeddings = model_output.pooler_output
embeddings = torch.nn.functional.normalize(embeddings)
print(embeddings)
```

The model has been truncated in [this notebook](https://colab.research.google.com/drive/1dnPRn0-ugj3vZgSpyCC9sgslM2SuSfHy?usp=sharing).
You can adapt it for other languages (like [EIStakovskii/LaBSE-fr-de](https://huggingface.co/EIStakovskii/LaBSE-fr-de)), models or datasets.

## Reference:
Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Narveen Ari, Wei Wang. [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852). July 2020

License: [https://tfhub.dev/google/LaBSE/1](https://tfhub.dev/google/LaBSE/1)
