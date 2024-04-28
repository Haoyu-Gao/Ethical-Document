---
language:
- en
license: apache-2.0
library_name: transformers
---

# Mega Masked LM on wikitext-103

This is the location on the Hugging Face hub for the Mega MLM checkpoint. I trained this model on the `wikitext-103` dataset using standard 
BERT-style masked LM pretraining using the [original Mega repository](https://github.com/facebookresearch/mega) and uploaded the weights 
initially to hf.co/mnaylor/mega-wikitext-103. When the implementation of Mega into Hugging Face's `transformers` is finished, the weights here
are designed to be used with `MegaForMaskedLM` and are compatible with the other (encoder-based) `MegaFor*` model classes.

This model uses the RoBERTa base tokenizer since the Mega paper does not implement a specific tokenizer aside from the character-level 
tokenizer used to illustrate long-sequence performance.