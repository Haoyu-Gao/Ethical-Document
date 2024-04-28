---
language: en
license: apache-2.0
---

# LongT5 (transient-global attention, large-sized model)

LongT5 model pre-trained on English language. The model was introduced in the paper [LongT5: Efficient Text-To-Text Transformer for Long Sequences](https://arxiv.org/pdf/2112.07916.pdf) by Guo et al. and first released in [the LongT5 repository](https://github.com/google-research/longt5). All the model architecture and configuration can be found in [Flaxformer repository](https://github.com/google/flaxformer) which uses another Google research project repository [T5x](https://github.com/google-research/t5x).

Disclaimer: The team releasing LongT5 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description
LongT5 model is an encoder-decoder transformer pre-trained in a text-to-text denoising generative setting ([Pegasus-like generation pre-training](https://arxiv.org/pdf/1912.08777.pdf)). LongT5 model is an extension of [T5 model](https://arxiv.org/pdf/1910.10683.pdf), and it enables using one of the two different efficient attention mechanisms - (1) Local attention, or (2) Transient-Global attention. The usage of attention sparsity patterns allows the model to efficiently handle input sequence.

LongT5 is particularly effective when fine-tuned for text generation (summarization, question answering) which requires handling long input sequences (up to 16,384 tokens).

Results of LongT5 (transient-global attention, large-sized model) fine-tuned on multiple (summarization, QA) tasks.

| Dataset | Rouge-1 | Rouge-2 | Rouge-Lsum |
| --- | --- | --- | --- |
| arXiv (16k input) | 48.28 | 21.63 | 44.11 |
| PubMed (16k input) | 49.98 | 24.69 | 46.46 |
| BigPatent (16k input) | 70.38 | 56.81 | 62.73 |
| MultiNews (8k input) | 47.18 | 18.44 | 24.18 |
| MediaSum (4k input) | 35.54 | 19.04 | 32.20 |
| CNN / DailyMail (4k input) | 42.49 | 20.51 | 40.18 |

| Dataset | EM | F1 |
| --- | --- | --- |
| Natural Questions (4k input) | 60.77 | 65.38 |
| Trivia QA (16k input) | 78.38 | 82.45 |

## Intended uses & limitations

The model is mostly meant to be fine-tuned on a supervised dataset. See the [model hub](https://huggingface.co/models?search=longt5) to look for fine-tuned versions on a task that interests you.

### How to use

```python
from transformers import AutoTokenizer, LongT5Model

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-large")
model = LongT5Model.from_pretrained("google/long-t5-tglobal-large")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

### BibTeX entry and citation info

```bibtex
@article{guo2021longt5,
  title={LongT5: Efficient Text-To-Text Transformer for Long Sequences},
  author={Guo, Mandy and Ainslie, Joshua and Uthus, David and Ontanon, Santiago and Ni, Jianmo and Sung, Yun-Hsuan and Yang, Yinfei},
  journal={arXiv preprint arXiv:2112.07916},
  year={2021}
}
```