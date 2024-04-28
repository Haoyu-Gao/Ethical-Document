---
license: mit
---

# ESMFold

ESMFold is a state-of-the-art end-to-end protein folding model based on an ESM-2 backbone. It does not require any lookup or MSA step, and therefore does not require any external databases to be present in order to make predictions. As a result, inference time is very significantly faster than AlphaFold2. For details on the model architecture and training, please refer to the [accompanying paper](https://www.science.org/doi/10.1126/science.ade2574).

If you're interested in using ESMFold in practice, please check out the associated [tutorial notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb).