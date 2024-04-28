---
inference: false
---
This is a model checkpoint for ["Should You Mask 15% in Masked Language Modeling"](https://arxiv.org/abs/2202.08005) [(code)](https://github.com/princeton-nlp/DinkyTrain.git).

The original checkpoint is avaliable at [princeton-nlp/efficient_mlm_m0.40](https://huggingface.co/princeton-nlp/efficient_mlm_m0.40). Unfortunately this checkpoint depends on code that isn't part of the official `transformers`
library. Additionally, the checkpoints contains unused weights due to a bug.

This checkpoint fixes the unused weights issue and uses the `RobertaPreLayerNorm` model from the `transformers`
library.

