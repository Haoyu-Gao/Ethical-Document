---
language: multilingual
widget:
- text: 🤗🤗🤗<mask>
- text: 🔥The goal of life is <mask> . 🔥
- text: Il segreto della vita è l’<mask> . ❤️
- text: Hasta <mask> 👋!
---


# Twitter-XLM-Roberta-base
This is a XLM-Roberta-base model trained on ~198M multilingual tweets, described and evaluated in the [reference paper](https://arxiv.org/abs/2104.12250). To evaluate this and other LMs on Twitter-specific data, please refer to the [main repository](https://github.com/cardiffnlp/xlm-t). A usage example is provided below. 

## Computing tweet similarity

```python
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_embedding(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    features = model(**encoded_input)
    features = features[0].detach().numpy() 
    features_mean = np.mean(features[0], axis=0) 
    return features_mean

query = "Acabo de pedir pollo frito 🐣" #spanish

tweets = ["We had a great time! ⚽️", # english
          "We hebben een geweldige tijd gehad! ⛩", # dutch
          "Nous avons passé un bon moment! 🎥", # french
          "Ci siamo divertiti! 🍝"] # italian

d = defaultdict(int)
for tweet in tweets:
    sim = 1-cosine(get_embedding(query),get_embedding(tweet))
    d[tweet] = sim
    
print('Most similar to: ',query)
print('----------------------------------------')
for idx,x in enumerate(sorted(d.items(), key=lambda x:x[1], reverse=True)):
  print(idx+1,x[0])
```
```
Most similar to:  Acabo de pedir pollo frito 🐣
----------------------------------------
1 Ci siamo divertiti! 🍝
2 Nous avons passé un bon moment! 🎥
3 We had a great time! ⚽️
4 We hebben een geweldige tijd gehad! ⛩
```

### BibTeX entry and citation info

Please cite the [reference paper](https://aclanthology.org/2022.lrec-1.27/) if you use this model.

```bibtex
@inproceedings{barbieri-etal-2022-xlm,
    title = "{XLM}-{T}: Multilingual Language Models in {T}witter for Sentiment Analysis and Beyond",
    author = "Barbieri, Francesco  and
      Espinosa Anke, Luis  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.27",
    pages = "258--266",
    abstract = "Language models are ubiquitous in current NLP, and their multilingual capacity has recently attracted considerable attention. However, current analyses have almost exclusively focused on (multilingual variants of) standard benchmarks, and have relied on clean pre-training and task-specific corpora as multilingual signals. In this paper, we introduce XLM-T, a model to train and evaluate multilingual language models in Twitter. In this paper we provide: (1) a new strong multilingual baseline consisting of an XLM-R (Conneau et al. 2020) model pre-trained on millions of tweets in over thirty languages, alongside starter code to subsequently fine-tune on a target task; and (2) a set of unified sentiment analysis Twitter datasets in eight different languages and a XLM-T model trained on this dataset.",
}
