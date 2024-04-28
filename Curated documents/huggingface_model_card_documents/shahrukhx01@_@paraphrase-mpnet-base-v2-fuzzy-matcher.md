---
tags:
- fuzzy-matching
- fuzzy-search
- entity-resolution
- record-linking
- structured-data-search
---
A Siamese BERT architecture trained at character levels tokens for embedding based Fuzzy matching.


## Usage (Sentence-Transformers)
Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:
```
pip install -U sentence-transformers
```
Then you can use the model like this:
```python
from sentence_transformers import SentenceTransformer, util
word1 = "fuzzformer"
word1 = " ".join([char for char in word1]) ## divide the word to char level to fuzzy match
word2 = "fizzformer"
word2 = " ".join([char for char in word2]) ## divide the word to char level to fuzzy match
words = [word1, word2]

model = SentenceTransformer('shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher')
fuzzy_embeddings = model.encode(words)

print("Fuzzy Match score:")
print(util.cos_sim(fuzzy_embeddings[0], fuzzy_embeddings[1]))
```
## Usage (HuggingFace Transformers)
```python
import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor, device

def cos_sim(a: Tensor, b: Tensor):
    """
    borrowed from sentence transformers repo
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Words we want fuzzy embeddings for
word1 = "fuzzformer"
word1 = " ".join([char for char in word1]) ## divide the word to char level to fuzzy match
word2 = "fizzformer"
word2 = " ".join([char for char in word2]) ## divide the word to char level to fuzzy match
words = [word1, word2]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher')
model = AutoModel.from_pretrained('shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher')

# Tokenize sentences
encoded_input = tokenizer(words, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
fuzzy_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Fuzzy Match score:")
print(cos_sim(fuzzy_embeddings[0], fuzzy_embeddings[1]))
```

## ACKNOWLEDGEMENT
A big thank you to [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) as their implementation really expedited the implementation of Fuzzformer.
## Citation
To cite FuzzTransformer in your work, please use the following bibtex reference:
@misc{shahrukhkhan2021fuzzTransformer, <br>
  author       = {Shahrukh Khan},<br>
  title        = {FuzzTransformer: A character level embedding based Siamese transformer for fuzzy string matching.},<br>
  year         = 2021,<br>
  publisher    = {Coming soon},<br>
  doi          = {Coming soon},<br>
  url          = {Coming soon}<br>
}
