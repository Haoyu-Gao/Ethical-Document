---
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
widget:
- source_sentence: '[*]CC[*]'
  sentences:
  - '[*]COC[*]'
  - '[*]CC(C)C[*]'
---

# kuelumbus/polyBERT

This is polyBERT: A chemical language model to enable fully machine-driven ultrafast polymer informatics. polyBERT maps PSMILES strings to 600 dimensional dense fingerprints. The fingerprints numerically represent polymer chemical structures. Please see the license agreement in the LICENSE file.

<!--- Describe your model here -->

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
psmiles_strings = ["[*]CC[*]", "[*]COC[*]"]

polyBERT = SentenceTransformer('kuelumbus/polyBERT')
embeddings = polyBERT.encode(psmiles_strings)
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
psmiles_strings = ["[*]CC[*]", "[*]COC[*]"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')

# Tokenize sentences
encoded_input = tokenizer(psmiles_strings, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = polyBERT(**encoded_input)

# Perform pooling. In this case, mean pooling.
fingerprints = mean_pooling(model_output, encoded_input['attention_mask'])

print("Fingerprints:")
print(fingerprints)
```



## Evaluation Results

See https://github.com/Ramprasad-Group/polyBERT and paper on arXiv.

## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: DebertaV2Model 
  (1): Pooling({'word_embedding_dimension': 600, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## Citing & Authors

Kuenneth, C., Ramprasad, R. polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics. Nat Commun 14, 4099 (2023). https://doi.org/10.1038/s41467-023-39868-6