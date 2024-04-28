---
license: mit
---

## Article Title Generator
The model is based on the T5 language model and trained using a large collection of Medium articles.

## Usage
Example code:
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
model = AutoModel.from_pretrained("czearing/article-title-generator")
```

## License
MIT

