---
language: en
tags:
- dpr
- dense-passage-retrieval
- knowledge-distillation
datasets:
- ms_marco
---

# Margin-MSE Trained ColBERT

We provide a retrieval trained DistilBert-based ColBERT model (https://arxiv.org/pdf/2004.12832.pdf). Our model is trained with Margin-MSE using a 3 teacher BERT_Cat (concatenated BERT scoring) ensemble on MSMARCO-Passage.

This instance can be used to **re-rank a candidate set** or **directly for a vector index based dense retrieval**. The architecure is a 6-layer DistilBERT, with an additional single linear layer at the end.

If you want to know more about our simple, yet effective knowledge distillation method for efficient information retrieval models for a variety of student architectures that is used for this model instance check out our paper: https://arxiv.org/abs/2010.02666 🎉

For more information, training data, source code, and a minimal usage example please visit: https://github.com/sebastian-hofstaetter/neural-ranking-kd

## Configuration

- fp16 trained, so fp16 inference shouldn't be a problem
- We use no compression: 768 dim output vectors (better suited for re-ranking, or storage for smaller collections, MSMARCO gets to ~1TB vector storage with fp16 ... ups)
- Query [MASK] augmention = 8x regardless of batch-size (needs to be added before the model, see the usage example in GitHub repo for more)

## Model Code

````python
from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch

class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True

class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)
        
        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor]):

        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document)

        score = self.forward_aggregation(query_vecs,document_vecs,query["attention_mask"],document["attention_mask"])
        return score

    def forward_representation(self,
                               tokens,
                               sequence_type=None) -> torch.Tensor:
        
        vecs = self.bert_model(**tokens)[0] # assuming a distilbert model here
        vecs = self.compressor(vecs)

        # if encoding only, zero-out the mask values so we can compress storage
        if sequence_type == "doc_encode" or sequence_type == "query_encode": 
            vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self,query_vecs, document_vecs,query_mask,document_mask):
        
        # create initial term-x-term scores (dot-product)
        score = torch.bmm(query_vecs, document_vecs.transpose(2,1))

        # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0 might select them)
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1,score.shape[1],-1)
        score[~exp_mask] = - 10000

        # max pooling over document dimension
        score = score.max(-1).values

        # mask out paddding query values
        score[~(query_mask.bool())] = 0

        # sum over query values
        score = score.sum(-1)

        return score

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # honestly not sure if that is the best way to go, but it works :)
model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")
````

## Effectiveness on MSMARCO Passage & TREC Deep Learning '19

We trained our model on the MSMARCO standard ("small"-400K query) training triples with knowledge distillation with a batch size of 32 on a single consumer-grade GPU (11GB memory).

For re-ranking we used the top-1000 BM25 results.

### MSMARCO-DEV

Here, we use the larger 49K query DEV set (same range as the smaller 7K DEV set, minimal changes possible)

|                                  | MRR@10 | NDCG@10 |
|----------------------------------|--------|---------|
| BM25                             | .194   | .241    |
| **Margin-MSE ColBERT** (Re-ranking) | .375   | .436   |

### TREC-DL'19

For MRR we use the recommended binarization point of the graded relevance of 2. This might skew the results when compared to other binarization point numbers.

|                                  | MRR@10 | NDCG@10 |
|----------------------------------|--------|---------|
| BM25                             | .689   | .501    |
| **Margin-MSE ColBERT** (Re-ranking) | .878   | .744    |

For more metrics, baselines, info and analysis, please see the paper: https://arxiv.org/abs/2010.02666

## Limitations & Bias

- The model inherits social biases from both DistilBERT and MSMARCO. 

- The model is only trained on relatively short passages of MSMARCO (avg. 60 words length), so it might struggle with longer text. 


## Citation

If you use our model checkpoint please cite our work as:

```
@misc{hofstaetter2020_crossarchitecture_kd,
      title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation}, 
      author={Sebastian Hofst{\"a}tter and Sophia Althammer and Michael Schr{\"o}der and Mete Sertkan and Allan Hanbury},
      year={2020},
      eprint={2010.02666},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```