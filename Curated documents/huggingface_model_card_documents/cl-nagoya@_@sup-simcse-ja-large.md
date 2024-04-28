---
language:
- ja
license: cc-by-sa-4.0
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
datasets:
- shunk031/jsnli
metrics:
- spearmanr
inference: false
---


# sup-simcse-ja-large


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U fugashi[unidic-lite] sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["こんにちは、世界！", "文埋め込み最高！文埋め込み最高と叫びなさい", "極度乾燥しなさい"]

model = SentenceTransformer("cl-nagoya/sup-simcse-ja-large")
embeddings = model.encode(sentences)
print(embeddings)
```



## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("cl-nagoya/sup-simcse-ja-large")
model = AutoModel.from_pretrained("cl-nagoya/sup-simcse-ja-large")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, cls pooling.
sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```

## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## Model Summary

- Fine-tuning method: Supervised SimCSE
- Base model: [cl-tohoku/bert-large-japanese-v2](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)
- Training dataset: [JSNLI](https://nlp.ist.i.kyoto-u.ac.jp/?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88)
- Pooling strategy: cls (with an extra MLP layer only during training)
- Hidden size: 1024
- Learning rate: 5e-5
- Batch size: 512
- Temperature: 0.05
- Max sequence length: 64
- Number of training examples: 2^20
- Validation interval (steps): 2^6
- Warmup ratio: 0.1
- Dtype: BFloat16

See the [GitHub repository](https://github.com/hppRC/simple-simcse-ja) for a detailed experimental setup.

## Citing & Authors

```
@misc{
  hayato-tsukagoshi-2023-simple-simcse-ja,
  author = {Hayato Tsukagoshi},
  title = {Japanese Simple-SimCSE},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hppRC/simple-simcse-ja}}
}
```