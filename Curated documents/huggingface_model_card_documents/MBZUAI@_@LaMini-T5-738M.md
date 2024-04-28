---
language:
- en
license: cc-by-nc-4.0
tags:
- generated_from_trainer
- instruction fine-tuning
pipeline_tag: text2text-generation
widget:
- text: how can I become more healthy?
  example_title: example
model-index:
- name: flan-t5-small-distil-v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

<p align="center" width="100%">
    <a><img src="https://raw.githubusercontent.com/mbzuai-nlp/lamini-lm/main/images/lamini.png" alt="Title" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# LaMini-T5-738M

[![Model License](https://img.shields.io/badge/Model%20License-CC%20By%20NC%204.0-red.svg)]()

This model is one of our LaMini-LM model series in paper "[LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions](https://github.com/mbzuai-nlp/lamini-lm)". This model is a fine-tuned version of [t5-large](https://huggingface.co/t5-large) on [LaMini-instruction dataset](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) that contains 2.58M samples for instruction fine-tuning. For more information about our dataset, please refer to our [project repository](https://github.com/mbzuai-nlp/lamini-lm/).  
You can view other models of LaMini-LM series as follows. Models with ✩ are those with the best overall performance given their size/architecture, hence we recommend using them. More details can be seen in our paper. 

<table>
<thead>
  <tr>
    <th>Base model</th>
    <th colspan="4">LaMini-LM series (#parameters)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>T5</td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-t5-61m" target="_blank" rel="noopener noreferrer">LaMini-T5-61M</a></td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-t5-223m" target="_blank" rel="noopener noreferrer">LaMini-T5-223M</a></td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-t5-738m" target="_blank" rel="noopener noreferrer">LaMini-T5-738M</a></td>
    <td></td>
  </tr>
   <tr>
        <td>Flan-T5</td>
        <td><a href="https://huggingface.co/MBZUAI/lamini-flan-t5-77m" target="_blank" rel="noopener noreferrer">LaMini-Flan-T5-77M</a>✩</td>
        <td><a href="https://huggingface.co/MBZUAI/lamini-flan-t5-248m" target="_blank" rel="noopener noreferrer">LaMini-Flan-T5-248M</a>✩</td>
        <td><a href="https://huggingface.co/MBZUAI/lamini-flan-t5-783m" target="_blank" rel="noopener noreferrer">LaMini-Flan-T5-783M</a>✩</td>
    <td></td>
  </tr>
    <tr>
    <td>Cerebras-GPT</td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-cerebras-111m" target="_blank" rel="noopener noreferrer">LaMini-Cerebras-111M</a></td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-cerebras-256m" target="_blank" rel="noopener noreferrer">LaMini-Cerebras-256M</a></td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-cerebras-590m" target="_blank" rel="noopener noreferrer">LaMini-Cerebras-590M</a></td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-cerebras-1.3b" target="_blank" rel="noopener noreferrer">LaMini-Cerebras-1.3B</a></td>
  </tr>
  <tr>
    <td>GPT-2</td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-gpt-124m" target="_blank" rel="noopener noreferrer">LaMini-GPT-124M</a>✩</td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-gpt-774m" target="_blank" rel="noopener noreferrer">LaMini-GPT-774M</a>✩</td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-gpt-1.5b" target="_blank" rel="noopener noreferrer">LaMini-GPT-1.5B</a>✩</td>
    <td></td>
  </tr>
  <tr>
    <td>GPT-Neo</td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-neo-125m" target="_blank" rel="noopener noreferrer">LaMini-Neo-125M</a></td>
    <td><a href="https://huggingface.co/MBZUAI/lamini-neo-1.3b" target="_blank" rel="noopener noreferrer">LaMini-Neo-1.3B</a></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GPT-J</td>
    <td colspan="4">coming soon</td>
  </tr>
  <tr>
    <td>LLaMA</td>
    <td colspan="4">coming soon</td>
  </tr>

  
</tbody>
</table>


## Use

### Intended use
We recommend using the model to response to human instructions written in natural language. 

We now show you how to load and use our model using HuggingFace `pipeline()`.

```python
# pip install -q transformers
from transformers import pipeline

checkpoint = "{model_name}"

model = pipeline('text2text-generation', model = checkpoint)

input_prompt = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'
generated_text = model(input_prompt, max_length=512, do_sample=True)[0]['generated_text']

print("Response", generated_text)
```

## Training Procedure

<p align="center" width="100%">
    <a><img src="https://raw.githubusercontent.com/mbzuai-nlp/lamini-lm/main/images/lamini-pipeline.drawio.png" alt="Title" style="width: 100%; min-width: 250px; display: block; margin: auto;"></a>
</p>

We initialize with [t5-large](https://huggingface.co/t5-large) and fine-tune it on our [LaMini-instruction dataset](https://huggingface.co/datasets/MBZUAI/LaMini-instruction). Its total number of parameters is 738M. 

### Training Hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 128
- eval_batch_size: 64
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 512
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

## Evaluation
We conducted two sets of evaluations: automatic evaluation on downstream NLP tasks and human evaluation on user-oriented instructions. For more detail, please refer to our [paper](). 

## Limitations

More information needed


# Citation

```bibtex
@article{lamini-lm,
  author       = {Minghao Wu and
                  Abdul Waheed and
                  Chiyu Zhang and
                  Muhammad Abdul-Mageed and
                  Alham Fikri Aji
                  },
  title        = {LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions},
  journal      = {CoRR},
  volume       = {abs/2304.14402},
  year         = {2023},
  url          = {https://arxiv.org/abs/2304.14402},
  eprinttype   = {arXiv},
  eprint       = {2304.14402}
}
```