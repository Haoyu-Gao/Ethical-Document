---
language: ru
tags:
- rubert
- russian
- nli
- rte
- zero-shot-classification
datasets:
- cointegrated/nli-rus-translated-v2021
pipeline_tag: zero-shot-classification
widget:
- text: Я хочу поехать в Австралию
  candidate_labels: спорт,путешествия,музыка,кино,книги,наука,политика
  hypothesis_template: Тема текста - {}.
---
# RuBERT for NLI (natural language inference)

This is the [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased) fine-tuned to predict the logical relationship between two short texts: entailment, contradiction, or neutral.

## Usage
How to run the model for NLI:
```python
# !pip install transformers sentencepiece --quiet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

text1 = 'Сократ - человек, а все люди смертны.'
text2 = 'Сократ никогда не умрёт.'
with torch.inference_mode():
    out = model(**tokenizer(text1, text2, return_tensors='pt').to(model.device))
    proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
print({v: proba[k] for k, v in model.config.id2label.items()})
# {'entailment': 0.009525929, 'contradiction': 0.9332064, 'neutral': 0.05726764} 
```

You can also use this model for zero-shot short text classification (by labels only), e.g. for sentiment analysis:

```python
def predict_zero_shot(text, label_texts, model, tokenizer, label='entailment', normalize=True):
    label_texts
    tokens = tokenizer([text] * len(label_texts), label_texts, truncation=True, return_tensors='pt', padding=True)
    with torch.inference_mode():
        result = torch.softmax(model(**tokens.to(model.device)).logits, -1)
    proba = result[:, model.config.label2id[label]].cpu().numpy()
    if normalize:
        proba /= sum(proba)
    return proba

classes = ['Я доволен', 'Я недоволен']
predict_zero_shot('Какая гадость эта ваша заливная рыба!', classes, model, tokenizer)
# array([0.05609814, 0.9439019 ], dtype=float32)
predict_zero_shot('Какая вкусная эта ваша заливная рыба!', classes, model, tokenizer)
# array([0.9059292 , 0.09407079], dtype=float32)
```

Alternatively, you can use [Huggingface pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) for inference.

## Sources
The model has been trained on a series of NLI datasets automatically translated to Russian from English.

Most datasets were taken [from the repo of Felipe Salvatore](https://github.com/felipessalvatore/NLI_datasets):
[JOCI](https://github.com/sheng-z/JOCI), 
[MNLI](https://cims.nyu.edu/~sbowman/multinli/), 
[MPE](https://aclanthology.org/I17-1011/), 
[SICK](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf), 
[SNLI](https://nlp.stanford.edu/projects/snli/).

Some datasets obtained from the original sources:
[ANLI](https://github.com/facebookresearch/anli), 
[NLI-style FEVER](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md),
[IMPPRES](https://github.com/facebookresearch/Imppres).

## Performance

The table below shows ROC AUC (one class vs rest) for five models on the corresponding *dev* sets:
- [tiny](https://huggingface.co/cointegrated/rubert-tiny-bilingual-nli): a small BERT predicting entailment vs not_entailment
- [twoway](https://huggingface.co/cointegrated/rubert-base-cased-nli-twoway): a base-sized BERT predicting entailment vs not_entailment
- [threeway](https://huggingface.co/cointegrated/rubert-base-cased-nli-threeway) (**this model**): a base-sized BERT predicting entailment vs contradiction vs neutral
- [vicgalle-xlm](https://huggingface.co/vicgalle/xlm-roberta-large-xnli-anli): a large multilingual NLI model
- [facebook-bart](https://huggingface.co/facebook/bart-large-mnli): a large multilingual NLI model


|model                   |add_one_rte|anli_r1|anli_r2|anli_r3|copa|fever|help|iie  |imppres|joci|mnli |monli|mpe |scitail|sick|snli|terra|total |
|------------------------|-----------|-------|-------|-------|----|-----|----|-----|-------|----|-----|-----|----|-------|----|----|-----|------|
|n_observations          |387        |1000   |1000   |1200   |200 |20474|3355|31232|7661   |939 |19647|269  |1000|2126   |500 |9831|307  |101128|
|tiny/entailment         |0.77       |0.59   |0.52   |0.53   |0.53|0.90 |0.81|0.78 |0.93   |0.81|0.82 |0.91 |0.81|0.78   |0.93|0.95|0.67 |0.77  |
|twoway/entailment       |0.89       |0.73   |0.61   |0.62   |0.58|0.96 |0.92|0.87 |0.99   |0.90|0.90 |0.99 |0.91|0.96   |0.97|0.97|0.87 |0.86  |
|threeway/entailment     |0.91       |0.75   |0.61   |0.61   |0.57|0.96 |0.56|0.61 |0.99   |0.90|0.91 |0.67 |0.92|0.84   |0.98|0.98|0.90 |0.80  |
|vicgalle-xlm/entailment |0.88       |0.79   |0.63   |0.66   |0.57|0.93 |0.56|0.62 |0.77   |0.80|0.90 |0.70 |0.83|0.84   |0.91|0.93|0.93 |0.78  |
|facebook-bart/entailment|0.51       |0.41   |0.43   |0.47   |0.50|0.74 |0.55|0.57 |0.60   |0.63|0.70 |0.52 |0.56|0.68   |0.67|0.72|0.64 |0.58  |
|threeway/contradiction  |           |0.71   |0.64   |0.61   |    |0.97 |    |     |1.00   |0.77|0.92 |     |0.89|       |0.99|0.98|     |0.85  |
|threeway/neutral        |           |0.79   |0.70   |0.62   |    |0.91 |    |     |0.99   |0.68|0.86 |     |0.79|       |0.96|0.96|     |0.83  |

For evaluation (and for training of the [tiny](https://huggingface.co/cointegrated/rubert-tiny-bilingual-nli) and [twoway](https://huggingface.co/cointegrated/rubert-base-cased-nli-twoway) models), some extra datasets were used: 
[Add-one RTE](https://cs.brown.edu/people/epavlick/papers/ans.pdf), 
[CoPA](https://people.ict.usc.edu/~gordon/copa.html), 
[IIE](https://aclanthology.org/I17-1100), and
[SCITAIL](https://allenai.org/data/scitail) taken from [the repo of Felipe Salvatore](https://github.com/felipessalvatore/NLI_datasets) and translatted,
[HELP](https://github.com/verypluming/HELP) and [MoNLI](https://github.com/atticusg/MoNLI) taken from the original sources and translated, 
and Russian [TERRa](https://russiansuperglue.com/ru/tasks/task_info/TERRa). 
