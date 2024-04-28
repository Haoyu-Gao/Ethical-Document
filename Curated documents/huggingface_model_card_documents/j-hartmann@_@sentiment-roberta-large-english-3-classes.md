---
language: en
tags:
- roberta
- sentiment
- twitter
widget:
- text: Oh no. This is bad..
- text: To be or not to be.
- text: Oh Happy Day
---

This RoBERTa-based model can classify the sentiment of English language text in 3 classes:

- positive 😀
- neutral 😐
- negative 🙁

The model was fine-tuned on 5,304 manually annotated social media posts. 
The hold-out accuracy is 86.1%. 
For details on the training approach see Web Appendix F in Hartmann et al. (2021). 

# Application
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes", return_all_scores=True)
classifier("This is so nice!")
```

```python
Output:
[[{'label': 'negative', 'score': 0.00016451838018838316},
  {'label': 'neutral', 'score': 0.000174045650055632},
  {'label': 'positive', 'score': 0.9996614456176758}]]
```

# Reference
Please cite [this paper](https://journals.sagepub.com/doi/full/10.1177/00222437211037258) when you use our model. Feel free to reach out to [jochen.hartmann@tum.de](mailto:jochen.hartmann@tum.de) with any questions or feedback you may have.
```
@article{hartmann2021,
  title={The Power of Brand Selfies},
  author={Hartmann, Jochen and Heitmann, Mark and Schamp, Christina and Netzer, Oded},
  journal={Journal of Marketing Research}
  year={2021}
}
```