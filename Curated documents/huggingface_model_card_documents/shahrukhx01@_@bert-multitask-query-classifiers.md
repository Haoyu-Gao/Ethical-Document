---
{}
---
# A Multi-task learning model with two prediction heads
* One prediction head classifies between keyword sentences vs statements/questions
* Other prediction head corresponds to classifier for statements vs questions

## Scores
##### Spaadia SQuaD Test acc: **0.9891**
##### Quora Keyword Pairs Test acc: **0.98048**

## Datasets:
Quora Keyword Pairs: https://www.kaggle.com/stefanondisponibile/quora-question-keyword-pairs
Spaadia SQuaD pairs: https://www.kaggle.com/shahrukhkhan/questions-vs-statementsclassificationdataset

## Article
[Medium article](https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf)
## Demo Notebook
[Colab Notebook Multi-task Query classifiers](https://colab.research.google.com/drive/1R7WcLHxDsVvZXPhr5HBgIWa3BlSZKY6p?usp=sharing)
## Clone the model repo
```bash 
git clone https://huggingface.co/shahrukhx01/bert-multitask-query-classifiers
```
```python
%cd bert-multitask-query-classifiers/
```
## Load model
```python
from multitask_model import BertForSequenceClassification
from transformers import AutoTokenizer
import torch
model = BertForSequenceClassification.from_pretrained(
        "shahrukhx01/bert-multitask-query-classifiers",
        task_labels_map={"quora_keyword_pairs": 2, "spaadia_squad_pairs": 2},
    )
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/bert-multitask-query-classifiers")
```
## Run inference on both Tasks
```python
from multitask_model import BertForSequenceClassification
from transformers import AutoTokenizer
import torch
model = BertForSequenceClassification.from_pretrained(
        "shahrukhx01/bert-multitask-query-classifiers",
        task_labels_map={"quora_keyword_pairs": 2, "spaadia_squad_pairs": 2},
    )
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/bert-multitask-query-classifiers")

## Keyword vs Statement/Question Classifier
input = ["keyword query", "is this a keyword query?"]
task_name="quora_keyword_pairs"
sequence = tokenizer(input, padding=True, return_tensors="pt")['input_ids']
logits = model(sequence, task_name=task_name)[0]
predictions = torch.argmax(torch.softmax(logits, dim=1).detach().cpu(), axis=1)
for input, prediction in zip(input, predictions):
  print(f"task: {task_name}, input: {input} \n prediction=> {prediction}")
  print()
  

## Statement vs Question Classifier
input = ["where is berlin?", "is this a keyword query?", "Berlin is in Germany."]
task_name="spaadia_squad_pairs"
sequence = tokenizer(input, padding=True, return_tensors="pt")['input_ids']
logits = model(sequence, task_name=task_name)[0]
predictions = torch.argmax(torch.softmax(logits, dim=1).detach().cpu(), axis=1)
for input, prediction in zip(input, predictions):
  print(f"task: {task_name}, input: {input} \n prediction=> {prediction}")
  print()
```