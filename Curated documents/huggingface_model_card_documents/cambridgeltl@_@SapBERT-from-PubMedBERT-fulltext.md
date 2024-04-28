---
language:
- en
license: apache-2.0
tags:
- biomedical
- lexical semantics
- bionlp
- biology
- science
- embedding
- entity linking
---
---


datasets:
- UMLS

**[news]** A cross-lingual extension of SapBERT will appear in the main onference of **ACL 2021**! <br>
**[news]** SapBERT will appear in the conference proceedings of **NAACL 2021**!

### SapBERT-PubMedBERT
SapBERT by [Liu et al. (2020)](https://arxiv.org/pdf/2010.11784.pdf). Trained with [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) 2020AA (English only), using [microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) as the base model.

### Expected input and output
The input should be a string of biomedical entity names, e.g., "covid infection" or "Hydroxychloroquine". The [CLS] embedding of the last layer is regarded as the output.

#### Extracting embeddings from SapBERT

The following script converts a list of strings (entity names) into embeddings.
```python
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel  

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()

# replace with your own list of entity names
all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 

bs = 128 # batch size during inference
all_embs = []
for i in tqdm(np.arange(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                       padding="max_length", 
                                       max_length=25, 
                                       truncation=True,
                                       return_tensors="pt")
    toks_cuda = {}
    for k,v in toks.items():
        toks_cuda[k] = v.cuda()
    cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
    all_embs.append(cls_rep.cpu().detach().numpy())

all_embs = np.concatenate(all_embs, axis=0)
```

For more details about training and eval, see SapBERT [github repo](https://github.com/cambridgeltl/sapbert).


### Citation
```bibtex
@inproceedings{liu-etal-2021-self,
    title = "Self-Alignment Pretraining for Biomedical Entity Representations",
    author = "Liu, Fangyu  and
      Shareghi, Ehsan  and
      Meng, Zaiqiao  and
      Basaldella, Marco  and
      Collier, Nigel",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.334",
    pages = "4228--4238",
    abstract = "Despite the widespread success of self-supervised learning via masked language models (MLM), accurately capturing fine-grained semantic relationships in the biomedical domain remains a challenge. This is of paramount importance for entity-level tasks such as entity linking where the ability to model entity relations (especially synonymy) is pivotal. To address this challenge, we propose SapBERT, a pretraining scheme that self-aligns the representation space of biomedical entities. We design a scalable metric learning framework that can leverage UMLS, a massive collection of biomedical ontologies with 4M+ concepts. In contrast with previous pipeline-based hybrid systems, SapBERT offers an elegant one-model-for-all solution to the problem of medical entity linking (MEL), achieving a new state-of-the-art (SOTA) on six MEL benchmarking datasets. In the scientific domain, we achieve SOTA even without task-specific supervision. With substantial improvement over various domain-specific pretrained MLMs such as BioBERT, SciBERTand and PubMedBERT, our pretraining scheme proves to be both effective and robust.",
}
```