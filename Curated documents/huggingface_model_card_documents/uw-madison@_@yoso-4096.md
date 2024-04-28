---
{}
---
# YOSO

YOSO model for masked language modeling (MLM) for sequence length 4096.

## About YOSO

The YOSO model was proposed in [You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://arxiv.org/abs/2111.09714) by Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh.

The abstract from the paper is the following:

Transformer-based models are widely used in natural language processing (NLP). Central to the transformer model is the self-attention mechanism, which captures the interactions of token pairs in the input sequences and depends quadratically on the sequence length. Training such models on longer sequences is expensive. In this paper, we show that a Bernoulli sampling attention mechanism based on Locality Sensitive Hashing (LSH), decreases the quadratic complexity of such models to linear. We bypass the quadratic cost by considering self-attention as a sum of individual tokens associated with Bernoulli random variables that can, in principle, be sampled at once by a single hash (although in practice, this number may be a small constant). This leads to an efficient sampling scheme to estimate self-attention which relies on specific modifications of LSH (to enable deployment on GPU architectures). We evaluate our algorithm on the GLUE benchmark with standard 512 sequence length where we see favorable performance relative to a standard pretrained Transformer. On the Long Range Arena (LRA) benchmark, for evaluating performance on long sequences, our method achieves results consistent with softmax self-attention but with sizable speed-ups and memory savings and often outperforms other efficient self-attention methods. Our code is available at this https URL

## Usage

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='uw-madison/yoso-4096')
>>> unmasker("Paris is the [MASK] of France.")

[{'score': 0.024274500086903572,
  'token': 812,
  'token_str': ' capital',
  'sequence': 'Paris is the capital of France.'},
 {'score': 0.022863076999783516,
  'token': 3497,
  'token_str': ' Republic',
  'sequence': 'Paris is the Republic of France.'},
 {'score': 0.01383623294532299,
  'token': 1515,
  'token_str': ' French',
  'sequence': 'Paris is the French of France.'},
 {'score': 0.013550693169236183,
  'token': 2201,
  'token_str': ' Paris',
  'sequence': 'Paris is the Paris of France.'},
 {'score': 0.011591030284762383,
  'token': 270,
  'token_str': ' President',
  'sequence': 'Paris is the President of France.'}]
```