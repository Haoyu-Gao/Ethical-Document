---
language:
- ru
tags:
- PyTorch
- Transformers
---

# BERT large model (uncased) for Sentence Embeddings in Russian language.
The model is described [in this article](https://habr.com/ru/company/sberdevices/blog/527576/)  
For better quality, use mean token embeddings.

## Usage (HuggingFace Models Repository)

You can use the model directly from the model repository to compute sentence embeddings:
```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



#Sentences we want sentence embeddings for
sentences = ['Привет! Как твои дела?',
             'А правда, что 42 твое любимое число?']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
```

# Authors
+ [SberDevices](https://sberdevices.ru/) Team.
+ Aleksandr Abramov: [HF profile](https://huggingface.co/Andrilko), [Github](https://github.com/Ab1992ao), [Kaggle Competitions Master](https://www.kaggle.com/andrilko);
+ Denis Antykhov: [Github](https://github.com/gaphex);