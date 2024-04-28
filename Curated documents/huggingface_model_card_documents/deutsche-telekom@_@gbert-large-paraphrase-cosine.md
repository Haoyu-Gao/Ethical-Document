---
language:
- de
license: mit
tags:
- sentence-transformers
- sentence-similarity
- transformers
- setfit
datasets:
- deutsche-telekom/ger-backtrans-paraphrase
pipeline_tag: sentence-similarity
---

# German BERT large paraphrase cosine
This is a [sentence-transformers](https://www.SBERT.net) model.
It maps sentences & paragraphs (text) into a 1024 dimensional dense vector space.
The model is intended to be used together with [SetFit](https://github.com/huggingface/setfit)
to improve German few-shot text classification.
It has a sibling model called
[deutsche-telekom/gbert-large-paraphrase-euclidean](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean).


This model is based on [deepset/gbert-large](https://huggingface.co/deepset/gbert-large).
Many thanks to [deepset](https://www.deepset.ai/)!

**Loss Function**\
We have used [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss)
with cosine similarity as the loss function.

**Training Data**\
The model is trained on a carefully filtered dataset of
[deutsche-telekom/ger-backtrans-paraphrase](https://huggingface.co/datasets/deutsche-telekom/ger-backtrans-paraphrase).
We deleted the following pairs of sentences:
- `min_char_len` less than 15
- `jaccard_similarity` greater than 0.3
- `de_token_count` greater than 30
- `en_de_token_count` greater than 30
- `cos_sim` less than 0.85

**Hyperparameters**
- learning_rate: 8.345726930229726e-06
- num_epochs: 7
- train_batch_size: 57
- num_gpu: 1

## Evaluation Results
We use the [NLU Few-shot Benchmark - English and German](https://huggingface.co/datasets/deutsche-telekom/NLU-few-shot-benchmark-en-de)
dataset to evaluate this model in a German few-shot scenario.

**Qualitative results**
- multilingual sentence embeddings provide the worst results
- Electra models also deliver poor results
- German BERT base size model ([deepset/gbert-base](https://huggingface.co/deepset/gbert-base)) provides good results
- German BERT large size model ([deepset/gbert-large](https://huggingface.co/deepset/gbert-large)) provides very good results
- our fine-tuned models (this model and [deutsche-telekom/gbert-large-paraphrase-euclidean](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean)) provide best results

## Licensing
Copyright (c) 2023 [Philip May](https://may.la/), [Deutsche Telekom AG](https://www.telekom.com/)\
Copyright (c) 2022 [deepset GmbH](https://www.deepset.ai/)

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-cosine/blob/main/LICENSE) in the repository.
