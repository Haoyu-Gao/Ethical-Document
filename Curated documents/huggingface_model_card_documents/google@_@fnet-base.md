---
language: en
license: apache-2.0
tags:
- fnet
datasets:
- c4
---

# FNet base model

Pretrained model on English language using a masked language modeling (MLM) and next sentence prediction (NSP) objective. It was 
introduced in [this paper](https://arxiv.org/abs/2105.03824) and first released in [this repository](https://github.com/google-research/google-research/tree/master/f_net).
This model is cased: it makes a difference between english and English. The model achieves 0.58 accuracy on MLM objective and 0.80 on NSP objective.

Disclaimer: This model card has been written by [gchhablani](https://huggingface.co/gchhablani).

## Model description

FNet is a transformers model with attention replaced with fourier transforms. Hence, the inputs do not contain an `attention_mask`. It is pretrained on a large corpus of 
English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling
them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and 
labels from those texts. More precisely, it was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes
  they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to
  predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard
classifier using the features produced by the FNet model as inputs.

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=fnet) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

## Training data

The FNet model was pretrained on [C4](https://huggingface.co/datasets/c4), a cleaned version of the Common Crawl dataset.

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using SentencePiece and a vocabulary size of 32,000. The inputs of the model are
then of the form:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

With probability 0.5, sentence A and sentence B correspond to two consecutive sentences in the original corpus and in
the other cases, it's another random sentence in the corpus. Note that what is considered a sentence here is a
consecutive span of text usually longer than a single sentence. The only constrain is that the result with the two
"sentences" has a combined length of less than 512 tokens.

The details of the masking procedure for each sentence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

### Pretraining

FNet-base was trained on 4 cloud TPUs in Pod configuration (16 TPU chips total) for one million steps with a batch size
of 256. The sequence length was limited to 512 tokens. The optimizer
used is Adam with a learning rate of 1e-4, \\(\beta_{1} = 0.9\\) and \\(\beta_{2} = 0.999\\), a weight decay of 0.01,
learning rate warmup for 10,000 steps and linear decay of the learning rate after.

## Evaluation results

FNet-base was fine-tuned and evaluated on the validation data of the [GLUE benchamrk](https://huggingface.co/datasets/glue). The results of the official model (written in Flax) can be seen in Table 1 on page 7 of [the official paper](https://arxiv.org/abs/2105.03824).

For comparison, this model (ported to PyTorch) was fine-tuned and evaluated using the [official Hugging Face GLUE evaluation scripts](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification#glue-tasks) alongside [bert-base-cased](https://hf.co/models/bert-base-cased) for comparison.
The training was done on a single 16GB NVIDIA Tesla V100 GPU. For MRPC/WNLI, the models were trained for 5 epochs, while for other tasks, the models were trained for 3 epochs. A sequence length of 512 was used with batch size 16 and learning rate 2e-5.

The following table summarizes the results for [fnet-base](https://huggingface.co/google/fnet-base) (called *FNet (PyTorch) - Reproduced*) and [bert-base-cased](https://hf.co/models/bert-base-cased) (called *Bert (PyTorch) - Reproduced*) in terms of **fine-tuning** speed. The format is *hour:min:seconds*. **Note** that the authors compared **pre-traning** speed in [the official paper](https://arxiv.org/abs/2105.03824) instead.

| Task/Model | FNet-base (PyTorch) |Bert-base (PyTorch)|
|:----:|:-----------:|:----:|
| MNLI-(m/mm) | [06:40:55](https://huggingface.co/gchhablani/fnet-base-finetuned-mnli) | [09:52:33](https://huggingface.co/gchhablani/bert-base-cased-finetuned-mnli)|
| QQP  | [06:21:16](https://huggingface.co/gchhablani/fnet-base-finetuned-qqp) | [09:25:01](https://huggingface.co/gchhablani/bert-base-cased-finetuned-qqp) |
| QNLI  | [01:48:22](https://huggingface.co/gchhablani/fnet-base-finetuned-qnli)  | [02:40:22](https://huggingface.co/gchhablani/bert-base-cased-finetuned-qnli)|
| SST-2 | [01:09:27](https://huggingface.co/gchhablani/fnet-base-finetuned-sst2) | [01:42:17](https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2)|
| CoLA  | [00:09:47](https://huggingface.co/gchhablani/fnet-base-finetuned-cola) | [00:14:20](https://huggingface.co/gchhablani/bert-base-cased-finetuned-cola)|
| STS-B | [00:07:09](https://huggingface.co/gchhablani/fnet-base-finetuned-stsb) | [00:10:24](https://huggingface.co/gchhablani/bert-base-cased-finetuned-stsb)|
| MRPC  | [00:07:48](https://huggingface.co/gchhablani/fnet-base-finetuned-mrpc) | [00:11:12](https://huggingface.co/gchhablani/bert-base-cased-finetuned-mrpc)|
| RTE  | [00:03:24](https://huggingface.co/gchhablani/fnet-base-finetuned-rte) | [00:04:51](https://huggingface.co/gchhablani/bert-base-cased-finetuned-rte)|
| WNLI | [00:02:37](https://huggingface.co/gchhablani/fnet-base-finetuned-wnli) | [00:03:23](https://huggingface.co/gchhablani/bert-base-cased-finetuned-wnli)|
| SUM | 16:30:45 | 24:23:56 |

On average the PyTorch version of FNet-base requires *ca.* 32% less time for GLUE fine-tuning on GPU.

The following table summarizes the results for [fnet-base](https://huggingface.co/google/fnet-base) (called *FNet (PyTorch) - Reproduced*) and [bert-base-cased](https://hf.co/models/bert-base-cased) (called *Bert (PyTorch) - Reproduced*) in terms of performance and compares it to the reported performance of the official FNet-base model (called *FNet (Flax) - Official*). Note that the training hyperparameters of the reproduced models were not the same as the official model, so the performance may differ significantly for some tasks (for example: CoLA). 

| Task/Model | Metric | FNet-base (PyTorch) | Bert-base (PyTorch) | FNet-Base (Flax - official) |
|:----:|:-----------:|:----:|:-----------:|:----:|
| MNLI-(m/mm) | Accuracy or Match/Mismatch  | [76.75](https://huggingface.co/gchhablani/fnet-base-finetuned-mnli) | [84.10](https://huggingface.co/gchhablani/bert-base-cased-finetuned-mnli) | 72/73 |
| QQP | mean(Accuracy,F1)  | [86.5](https://huggingface.co/gchhablani/fnet-base-finetuned-qqp) | [89.26](https://huggingface.co/gchhablani/bert-base-cased-finetuned-qqp) | 83 |
| QNLI | Accuracy  | [84.39](https://huggingface.co/gchhablani/fnet-base-finetuned-qnli) | [90.99](https://huggingface.co/gchhablani/bert-base-cased-finetuned-qnli) | 80 | 
| SST-2 | Accuracy  | [89.45](https://huggingface.co/gchhablani/fnet-base-finetuned-sst2) | [92.32](https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2) | 95 |
| CoLA | Matthews corr or Accuracy | [35.94](https://huggingface.co/gchhablani/fnet-base-finetuned-cola) | [59.57](https://huggingface.co/gchhablani/bert-base-cased-finetuned-cola) | 69 |
| STS-B | Spearman corr. | [82.19](https://huggingface.co/gchhablani/fnet-base-finetuned-stsb) | [88.98](https://huggingface.co/gchhablani/bert-base-cased-finetuned-stsb) | 79 | 
| MRPC | mean(F1/Accuracy) | [81.15](https://huggingface.co/gchhablani/fnet-base-finetuned-mrpc) | [88.15](https://huggingface.co/gchhablani/bert-base-cased-finetuned-mrpc) | 76 | 
| RTE | Accuracy | [62.82](https://huggingface.co/gchhablani/fnet-base-finetuned-rte) | [67.15](https://huggingface.co/gchhablani/bert-base-cased-finetuned-rte) | 63 |
| WNLI | Accuracy | [54.93](https://huggingface.co/gchhablani/fnet-base-finetuned-wnli) | [46.48](https://huggingface.co/gchhablani/bert-base-cased-finetuned-wnli) | - |
| Avg  | - | 72.7 | 78.6 | 76.7 |

We can see that FNet-base achieves around 93% of BERT-base's performance on average.

For more details, please refer to the checkpoints linked with the scores. On overview of all fine-tuned checkpoints of the following table can be accessed [here](https://huggingface.co/models?other=fnet-bert-base-comparison).
 
### How to use

You can use this model directly with a pipeline for masked language modeling:

**Note: The mask filling pipeline doesn't work exactly as the original model performs masking after converting to tokens. In masking pipeline an additional space is added after the [MASK].**

```python
>>> from transformers import FNetForMaskedLM, FNetTokenizer, pipeline
>>> tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")
>>> model = FNetForMaskedLM.from_pretrained("google/fnet-base")
>>> unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
>>> unmasker("Hello I'm a [MASK] model.")

[
    {"sequence": "hello i'm a new model.", "score": 0.12073223292827606, "token": 351, "token_str": "new"},
    {"sequence": "hello i'm a first model.", "score": 0.08501081168651581, "token": 478, "token_str": "first"},
    {"sequence": "hello i'm a next model.", "score": 0.060546260327100754, "token": 1037, "token_str": "next"},
    {"sequence": "hello i'm a last model.", "score": 0.038265593349933624, "token": 813, "token_str": "last"},
    {"sequence": "hello i'm a sister model.", "score": 0.033868927508592606, "token": 6232, "token_str": "sister"},
]

```

Here is how to use this model to get the features of a given text in PyTorch:

**Note: You must specify the maximum sequence length to be 512 and truncate/pad to the same length because the original model has no attention mask and considers all the hidden states during forward pass.**

```python
from transformers import FNetTokenizer, FNetModel
tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")
model = FNetModel.from_pretrained("google/fnet-base")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
output = model(**encoded_input)
```

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2105-03824,
  author    = {James Lee{-}Thorp and
               Joshua Ainslie and
               Ilya Eckstein and
               Santiago Onta{\~{n}}{\'{o}}n},
  title     = {FNet: Mixing Tokens with Fourier Transforms},
  journal   = {CoRR},
  volume    = {abs/2105.03824},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.03824},
  archivePrefix = {arXiv},
  eprint    = {2105.03824},
  timestamp = {Fri, 14 May 2021 12:13:30 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-03824.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Contributions
Thanks to [@gchhablani](https://huggingface.co/gchhablani) for adding this model.