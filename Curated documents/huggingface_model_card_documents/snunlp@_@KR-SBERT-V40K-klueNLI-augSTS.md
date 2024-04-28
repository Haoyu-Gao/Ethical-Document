---
language:
- ko
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
widget:
- source_sentence: 그 식당은 파리를 날린다
  sentences:
  - 그 식당에는 손님이 없다
  - 그 식당에서는 드론을 날린다
  - 파리가 식당에 날아다닌다
  example_title: Restaurant
- source_sentence: 잠이 옵니다
  sentences:
  - 잠이 안 옵니다
  - 졸음이 옵니다
  - 기차가 옵니다
  example_title: Sleepy
---

# snunlp/KR-SBERT-V40K-klueNLI-augSTS

This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

<!--- Describe your model here -->

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embeddings = model.encode(sentences)
print(embeddings)
```



## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
model = AutoModel.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```



## Evaluation Results

<!--- Describe how your model was evaluated -->

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=snunlp/KR-SBERT-V40K-klueNLI-augSTS)



## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## Application for document classification

Tutorial in Google Colab: https://colab.research.google.com/drive/1S6WSjOx9h6Wh_rX1Z2UXwx9i_uHLlOiM


|Model|Accuracy|
|-|-|
|KR-SBERT-Medium-NLI-STS|0.8400|
|KR-SBERT-V40K-NLI-STS|0.8400|
|KR-SBERT-V40K-NLI-augSTS|0.8511|
|KR-SBERT-V40K-klueNLI-augSTS|**0.8628**|


## Citation

```bibtex
@misc{kr-sbert,
  author = {Park, Suzi and Hyopil Shin},
  title = {KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snunlp/KR-SBERT}}
}
```