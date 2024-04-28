---
language: en
license: apache-2.0
datasets:
- c4
- wikipedia
- natural_questions
pipeline_tag: text2text-generation
---

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) for **Closed Book Question Answering**.

The model was pre-trained using T5's denoising objective on [C4](https://huggingface.co/datasets/c4), subsequently additionally pre-trained using [REALM](https://arxiv.org/pdf/2002.08909.pdf)'s salient span masking objective on [Wikipedia](https://huggingface.co/datasets/wikipedia), and finally fine-tuned on [Natural Questions (NQ)](https://huggingface.co/datasets/natural_questions).

**Note**: The model was fine-tuned on 100% of the train splits of [Natural Questions (NQ)](https://huggingface.co/datasets/natural_questions) for 10k steps.

Other community Checkpoints: [here](https://huggingface.co/models?search=ssm)

Paper: [How Much Knowledge Can You Pack
Into the Parameters of a Language Model?](https://arxiv.org/abs/1910.10683.pdf)

Authors: *Adam Roberts, Colin Raffel, Noam Shazeer* 

## Results on Natural Questions - Test Set

|Id | link | Exact Match  |
|---|---|---|
|T5-small|https://huggingface.co/google/t5-small-ssm-nq|25.5|
|**T5-large**|**https://huggingface.co/google/t5-large-ssm-nq**|**30.4**|
|T5-xl|https://huggingface.co/google/t5-xl-ssm-nq|35.6|
|T5-xxl|https://huggingface.co/google/t5-xxl-ssm-nq|37.9|
|T5-3b|https://huggingface.co/google/t5-3b-ssm-nq|33.2|
|T5-11b|https://huggingface.co/google/t5-11b-ssm-nq|36.6|

## Usage

The model can be used as follows for **closed book question answering**:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-large-ssm-nq")
t5_tok = AutoTokenizer.from_pretrained("google/t5-large-ssm-nq")

input_ids = t5_tok("When was Franklin D. Roosevelt born?", return_tensors="pt").input_ids
gen_output = t5_qa_model.generate(input_ids)[0]

print(t5_tok.decode(gen_output, skip_special_tokens=True))

# should give "December 26, 1892" => close, but not correct.
```

## Abstract

It has recently been observed that neural language models trained on unstructured text can implicitly store and retrieve knowledge using natural language queries. In this short paper, we measure the practical utility of this approach by fine-tuning pre-trained models to answer questions without access to any external context or knowledge. We show that this approach scales with model size and performs competitively with open-domain systems that explicitly retrieve answers from an external knowledge source when answering questions. To facilitate reproducibility and future work, we release our code and trained models at https://goo.gle/t5-cbqa.

![model image](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/how_much_know_ledge_image.png)