---
language:
- en
license: bsd-3-clause-clear
datasets:
- cnn_dailymail
metrics:
- f1
---
# FactCC factuality prediction model

Original paper: [Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)

This is a more modern implementation of the model and code from [the original github repo](https://github.com/salesforce/factCC)

This model is trained to predict whether a summary is factual with respect to the original text. Basic usage:
```
from transformers import BertForSequenceClassification, BertTokenizer
model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

text='''The US has "passed the peak" on new coronavirus cases, the White House reported. They predict that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.'''
wrong_summary = '''The pandemic has almost not affected the US'''

input_dict = tokenizer(text, wrong_summary, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
logits = model(**input_dict).logits
pred = logits.argmax(dim=1)
model.config.id2label[pred.item()] # prints: INCORRECT
```

It can also be used with a pipeline. Beware that since pipelines are not thought to be used with pair of sentences, and you have to use this double-list hack:
```
>>> from transformers import pipeline

>>> pipe=pipeline(model="manueldeprada/FactCC")
>>> pipe([[[text1,summary1]],[[text2,summary2]]],truncation='only_first',padding='max_length')
# output [{'label': 'INCORRECT', 'score': 0.9979124665260315}, {'label': 'CORRECT', 'score': 0.879124665260315}]
```

Example on how to perform batched inference to reproduce authors results on the test set:
```
def batched_FactCC(text_l, summary_l, max_length=512):    
    input_dict = tokenizer(text_l, summary_l, max_length=max_length, padding='max_length', truncation='only_first', return_tensors='pt')
    with torch.no_grad():
        logits = model(**input_dict).logits
        preds = logits.argmax(dim=1)
        return logits, preds

texts = []
claims = []
labels = []
with open('factCC/annotated_data/test/data-dev.jsonl', 'r') as file:
    for line in file:
        obj = json.loads(line)  # Load the JSON data from each line
        texts.append(obj['text'])
        claims.append(obj['claim'])
        labels.append(model.config.label2id[o['label']])

preds = []
batch_size = 8
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    batch_claims = claims[i:i+batch_size]
    _, preds = fact_cc(batch_texts, batch_claims)
    preds.extend(preds.tolist())

print(f"F1 micro: {f1_score(labels, preds, average='micro')}")
print(f"Balanced accuracy: {balanced_accuracy_score(labels, preds)}")
```