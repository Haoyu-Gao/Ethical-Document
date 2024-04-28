---
language:
- en
license: mit
library_name: transformers
tags:
- text-classification
- zero-shot-classification
pipeline_tag: zero-shot-classification
---

# deberta-v3-large-zeroshot-v1
## Model description
The model is designed for zero-shot classification with the Hugging Face pipeline. 
The model should be substantially better at zero-shot classification than my other zero-shot models on the
Hugging Face hub: https://huggingface.co/MoritzLaurer.

The model can do one universal task: determine whether a hypothesis is `true` or `not_true`
given a text (also called `entailment` vs. `not_entailment`).  
This task format is based on the Natural Language Inference task (NLI).
The task is so universal that any classification task can be reformulated into the task.

## Training data
The model was trained on a mixture of 27 tasks and 310 classes that have been reformatted into this universal format.
1.   26 classification tasks with ~400k texts:
'amazonpolarity', 'imdb', 'appreviews', 'yelpreviews', 'rottentomatoes',
'emotiondair', 'emocontext', 'empathetic',
'financialphrasebank', 'banking77', 'massive',
'wikitoxic_toxicaggregated', 'wikitoxic_obscene', 'wikitoxic_threat', 'wikitoxic_insult', 'wikitoxic_identityhate', 
'hateoffensive', 'hatexplain', 'biasframes_offensive', 'biasframes_sex', 'biasframes_intent',
'agnews', 'yahootopics',
'trueteacher', 'spam', 'wellformedquery'.
See details on each dataset here: https://docs.google.com/spreadsheets/d/1Z18tMh02IiWgh6o8pfoMiI_LH4IXpr78wd_nmNd5FaE/edit?usp=sharing
3.   Five NLI datasets with ~885k texts: "mnli", "anli", "fever", "wanli", "ling"

Note that compared to other NLI models, this model predicts two classes (`entailment` vs. `not_entailment`)
as opposed to three classes (entailment/neutral/contradiction)


### How to use the model
#### Simple zero-shot classification pipeline
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1")
sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
```

### Details on data and training
The code for preparing the data and training & evaluating the model is fully open-source here: https://github.com/MoritzLaurer/zeroshot-classifier/tree/main

## Limitations and bias
The model can only do text classification tasks. 

Please consult the original DeBERTa paper and the papers for the different datasets for potential biases. 

## License
The base model (DeBERTa-v3) is published under the MIT license.
The datasets the model was fine-tuned on are published under a diverse set of licenses.
The following spreadsheet provides an overview of the non-NLI datasets used for fine-tuning.
The spreadsheets contains information on licenses, the underlying papers etc.: https://docs.google.com/spreadsheets/d/1Z18tMh02IiWgh6o8pfoMiI_LH4IXpr78wd_nmNd5FaE/edit?usp=sharing

In addition, the model was also trained on the following NLI datasets: MNLI, ANLI, WANLI, LING-NLI, FEVER-NLI.

## Citation
If you use this model, please cite: 
```
@article{laurer_less_2023,
	title = {Less {Annotating}, {More} {Classifying}: {Addressing} the {Data} {Scarcity} {Issue} of {Supervised} {Machine} {Learning} with {Deep} {Transfer} {Learning} and {BERT}-{NLI}},
	issn = {1047-1987, 1476-4989},
	shorttitle = {Less {Annotating}, {More} {Classifying}},
	url = {https://www.cambridge.org/core/product/identifier/S1047198723000207/type/journal_article},
	doi = {10.1017/pan.2023.20},
	language = {en},
	urldate = {2023-06-20},
	journal = {Political Analysis},
	author = {Laurer, Moritz and Van Atteveldt, Wouter and Casas, Andreu and Welbers, Kasper},
	month = jun,
	year = {2023},
	pages = {1--33},
}
```

### Ideas for cooperation or questions?
If you have questions or ideas for cooperation, contact me at m{dot}laurer{at}vu{dot}nl or [LinkedIn](https://www.linkedin.com/in/moritz-laurer/)

### Debugging and issues
Note that DeBERTa-v3 was released on 06.12.21 and older versions of HF Transformers seem to have issues running the model (e.g. resulting in an issue with the tokenizer). Using Transformers>=4.13 might solve some issues.