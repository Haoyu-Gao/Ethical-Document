---
language: pt
license: apache-2.0
widget:
- text: O futuro de DI caiu 20 bps nesta manhã
  example_title: Example 1
- text: O Nubank decidiu cortar a faixa de preço da oferta pública inicial (IPO) após
    revés no humor dos mercados internacionais com as fintechs.
  example_title: Example 2
- text: O Ibovespa acompanha correção do mercado e fecha com alta moderada
  example_title: Example 3
---

# FinBERT-PT-BR : Financial BERT PT BR

FinBERT-PT-BR is a pre-trained NLP model to analyze sentiment of Brazilian Portuguese financial texts.

The model was trained in two main stages: language modeling and sentiment modeling. In the first stage, a language model was trained with more than 1.4 million texts of financial news in Portuguese. 
From this first training, it was possible to build a sentiment classifier with few labeled texts (500) that presented a satisfactory convergence.

At the end of the work, a comparative analysis with other models and the possible applications of the developed model are presented. 
In the comparative analysis, it was possible to observe that the developed model presented better results than the current models in the state of the art. 
Among the applications, it was demonstrated that the model can be used to build sentiment indices, investment strategies and macroeconomic data analysis, such as inflation.

## Applications

### Sentiment Index

![Sentiment Index](sentiment_index_and_economy.png)


## Usage

#### BertForSequenceClassification

```python
from transformers import AutoTokenizer, BertForSequenceClassification
import numpy as np
  
pred_mapper = {
    0: "POSITIVE",
    1: "NEGATIVE",
    2: "NEUTRAL"
  }

tokenizer = AutoTokenizer.from_pretrained("lucas-leme/FinBERT-PT-BR")
finbertptbr = BertForSequenceClassification.from_pretrained("lucas-leme/FinBERT-PT-BR")

tokens = tokenizer(["Hoje a bolsa caiu", "Hoje a bolsa subiu"], return_tensors="pt",
                    padding=True, truncation=True, max_length=512)
finbertptbr_outputs = finbertptbr(**tokens)

preds = [pred_mapper[np.argmax(pred)] for pred in finbertptbr_outputs.logits.cpu().detach().numpy()]
```

#### Pipeline

```python
from transformers import (
    AutoTokenizer, 
    BertForSequenceClassification,
    pipeline,
)

finbert_pt_br_tokenizer = AutoTokenizer.from_pretrained("lucas-leme/FinBERT-PT-BR")
finbert_pt_br_model = BertForSequenceClassification.from_pretrained("lucas-leme/FinBERT-PT-BR")

finbert_pt_br_pipeline = pipeline(task='text-classification', model=finbert_pt_br_model, tokenizer=finbert_pt_br_tokenizer)
finbert_pt_br_pipeline(['Hoje a bolsa caiu', 'Hoje a bolsa subiu'])
```

## Author

  - [Lucas Leme](https://www.linkedin.com/in/lucas-leme-santos/) - lucaslssantos99@gmail.com

## Paper

- Paper: [FinBERT-PT-BR: Sentiment Analysis of Texts in Portuguese from the Financial Market](https://sol.sbc.org.br/index.php/bwaif/article/view/24960)
- Undergraduate thesis: [FinBERT-PT-BR: Análise de sentimentos de textos em português referentes ao mercado financeiro](https://pcs.usp.br/pcspf/wp-content/uploads/sites/8/2022/12/Monografia_PCS3860_COOP_2022_Grupo_C12.pdf) 
