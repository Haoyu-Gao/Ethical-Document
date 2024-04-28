---
language: en
license: mit
widget:
- text: COVID-19 is
---

## BioGPT

Pre-trained language models have attracted increasing attention in the biomedical domain, inspired by their great success in the general natural language domain. Among the two main branches of pre-trained language models in the general language domain, i.e. BERT (and its variants) and GPT (and its variants), the first one has been extensively studied in the biomedical domain, such as BioBERT and PubMedBERT. While they have achieved great success on a variety of discriminative downstream biomedical tasks, the lack of generation ability constrains their application scope. In this paper, we propose BioGPT, a domain-specific generative Transformer language model pre-trained on large-scale biomedical literature. We evaluate BioGPT on six biomedical natural language processing tasks and demonstrate that our model outperforms previous models on most tasks. Especially, we get 44.98%, 38.42% and 40.76% F1 score on BC5CDR, KD-DTI and DDI end-to-end relation extraction tasks, respectively, and 78.2% accuracy on PubMedQA, creating a new record. Our case study on text generation further demonstrates the advantage of BioGPT on biomedical literature to generate fluent descriptions for biomedical terms.

You can use this model directly with a pipeline for text generation. Since the generation relies on some randomness, we
set a seed for reproducibility:

```python
>>> from transformers import pipeline, set_seed
>>> from transformers import BioGptTokenizer, BioGptForCausalLM
>>> model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
>>> tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
>>> generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
>>> set_seed(42)
>>> generator("COVID-19 is", max_length=20, num_return_sequences=5, do_sample=True)
[{'generated_text': 'COVID-19 is a disease that spreads worldwide and is currently found in a growing proportion of the population'},
 {'generated_text': 'COVID-19 is one of the largest viral epidemics in the world.'},
 {'generated_text': 'COVID-19 is a common condition affecting an estimated 1.1 million people in the United States alone.'},
 {'generated_text': 'COVID-19 is a pandemic, the incidence has been increased in a manner similar to that in other'},
 {'generated_text': 'COVID-19 is transmitted via droplets, air-borne, or airborne transmission.'}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BioGptTokenizer, BioGptForCausalLM
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

Beam-search decoding:

```python
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

sentence = "COVID-19 is"
inputs = tokenizer(sentence, return_tensors="pt")

set_seed(42)

with torch.no_grad():
    beam_output = model.generate(**inputs,
                                min_length=100,
                                max_length=1024,
                                num_beams=5,
                                early_stopping=True
                                )
tokenizer.decode(beam_output[0], skip_special_tokens=True)
'COVID-19 is a global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of coronavirus disease 2019 (COVID-19), which has spread to more than 200 countries and territories, including the United States (US), Canada, Australia, New Zealand, the United Kingdom (UK), and the United States of America (USA), as of March 11, 2020, with more than 800,000 confirmed cases and more than 800,000 deaths.'
```

## Citation

If you find BioGPT useful in your research, please cite the following paper:

```latex
@article{10.1093/bib/bbac409,
    author = {Luo, Renqian and Sun, Liai and Xia, Yingce and Qin, Tao and Zhang, Sheng and Poon, Hoifung and Liu, Tie-Yan},
    title = "{BioGPT: generative pre-trained transformer for biomedical text generation and mining}",
    journal = {Briefings in Bioinformatics},
    volume = {23},
    number = {6},
    year = {2022},
    month = {09},
    abstract = "{Pre-trained language models have attracted increasing attention in the biomedical domain, inspired by their great success in the general natural language domain. Among the two main branches of pre-trained language models in the general language domain, i.e. BERT (and its variants) and GPT (and its variants), the first one has been extensively studied in the biomedical domain, such as BioBERT and PubMedBERT. While they have achieved great success on a variety of discriminative downstream biomedical tasks, the lack of generation ability constrains their application scope. In this paper, we propose BioGPT, a domain-specific generative Transformer language model pre-trained on large-scale biomedical literature. We evaluate BioGPT on six biomedical natural language processing tasks and demonstrate that our model outperforms previous models on most tasks. Especially, we get 44.98\%, 38.42\% and 40.76\% F1 score on BC5CDR, KD-DTI and DDI end-to-end relation extraction tasks, respectively, and 78.2\% accuracy on PubMedQA, creating a new record. Our case study on text generation further demonstrates the advantage of BioGPT on biomedical literature to generate fluent descriptions for biomedical terms.}",
    issn = {1477-4054},
    doi = {10.1093/bib/bbac409},
    url = {https://doi.org/10.1093/bib/bbac409},
    note = {bbac409},
    eprint = {https://academic.oup.com/bib/article-pdf/23/6/bbac409/47144271/bbac409.pdf},
}
```
