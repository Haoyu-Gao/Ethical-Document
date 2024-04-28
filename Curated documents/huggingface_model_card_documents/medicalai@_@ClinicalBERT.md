---
tags:
- medical
---
# ClinicalBERT

<!-- Provide a quick summary of what the model is/does. -->

This model card describes the ClinicalBERT model, which was trained on a large multicenter dataset with a large corpus of 1.2B words of diverse diseases we constructed.
We then utilized a large-scale corpus of EHRs from over 3 million patient records to fine tune the base language model.

## Pretraining Data

The ClinicalBERT model was trained on a large multicenter dataset with a large corpus of 1.2B words of diverse diseases we constructed.
<!-- For more details, see here.  -->

## Model Pretraining

### Pretraining Procedures
The ClinicalBERT was initialized from BERT. Then the training followed the principle of masked language model, in which given a piece of text, we randomly replace some tokens by MASKs, 
special tokens for masking, and then require the model to predict the original tokens via contextual text. 

### Pretraining Hyperparameters

We used a batch size of 32, a maximum sequence length of 256, and a learning rate of 5e-5 for pre-training our models. 

## How to use the model

Load the model via the transformers library:
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
```

## Citation

Please cite this article:  Wang, G., Liu, X., Ying, Z. et al. Optimized glycemic control of type 2 diabetes with reinforcement learning: a proof-of-concept trial. Nat Med (2023). https://doi.org/10.1038/s41591-023-02552-9
