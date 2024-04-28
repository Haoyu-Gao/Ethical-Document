---
language:
- en
license: mit
tags:
- punctuation
datasets:
- yelp_polarity
metrics:
- f1
---
# ✨ bert-restore-punctuation
[![forthebadge](https://forthebadge.com/images/badges/gluten-free.svg)]()

This a bert-base-uncased model finetuned for punctuation restoration on [Yelp Reviews](https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews). 

The model predicts the punctuation and upper-casing of plain, lower-cased text. An example use case can be ASR output. Or other cases when text has lost punctuation.

This model is intended for direct use as a punctuation restoration model for the general English language. Alternatively, you can use this for further fine-tuning on domain-specific texts for punctuation restoration tasks.

Model restores the following punctuations -- **[! ? . , - : ; ' ]**

The model also restores the upper-casing of words.

-----------------------------------------------
## 🚋 Usage
**Below is a quick way to get up and running with the model.**
1. First, install the package.
```bash
pip install rpunct
```
2. Sample python code.
```python
from rpunct import RestorePuncts
# The default language is 'english'
rpunct = RestorePuncts()
rpunct.punctuate("""in 2018 cornell researchers built a high-powered detector that in combination with an algorithm-driven process called ptychography set a world record
by tripling the resolution of a state-of-the-art electron microscope as successful as it was that approach had a weakness it only worked with ultrathin samples that were
a few atoms thick anything thicker would cause the electrons to scatter in ways that could not be disentangled now a team again led by david muller the samuel b eckert
professor of engineering has bested its own record by a factor of two with an electron microscope pixel array detector empad that incorporates even more sophisticated
3d reconstruction algorithms the resolution is so fine-tuned the only blurring that remains is the thermal jiggling of the atoms themselves""")
# Outputs the following:
# In 2018, Cornell researchers built a high-powered detector that, in combination with an algorithm-driven process called Ptychography, set a world record by tripling the
# resolution of a state-of-the-art electron microscope. As successful as it was, that approach had a weakness. It only worked with ultrathin samples that were a few atoms
# thick. Anything thicker would cause the electrons to scatter in ways that could not be disentangled. Now, a team again led by David Muller, the Samuel B. 
# Eckert Professor of Engineering, has bested its own record by a factor of two with an Electron microscope pixel array detector empad that incorporates even more
# sophisticated 3d reconstruction algorithms. The resolution is so fine-tuned the only blurring that remains is the thermal jiggling of the atoms themselves.
```

**This model works on arbitrarily large text in English language and uses GPU if available.**

-----------------------------------------------
## 📡 Training data

Here is the number of product reviews we used for finetuning the model:

| Language | Number of text samples|
| -------- | ----------------- |
| English  | 560,000           |

We found the best convergence around _**3 epochs**_, which is what presented here and available via a download.

-----------------------------------------------
## 🎯 Accuracy
The fine-tuned model obtained the following accuracy on 45,990 held-out text samples:

| Accuracy | Overall F1 | Eval Support |
| -------- | ---------------------- | ------------------- |
| 91%  | 90%                 | 45,990

Below is a breakdown of the performance of the model by each label:

|  label    |   precision  |  recall | f1-score  | support|
| --------- | -------------|-------- | ----------|--------|
|     **!**    |   0.45       | 0.17    |  0.24     |  424   
|     **!+Upper**    |   0.43       | 0.34    |  0.38     |   98   
|     **'**    |   0.60       | 0.27    |  0.37     |   11   
|    **,**    |   0.59       | 0.51    |  0.55     | 1522   
|     **,+Upper**    |   0.52       | 0.50    |  0.51     |  239   
|     **-**    |   0.00       | 0.00    |  0.00     |   18   
|     **.**    |   0.69       | 0.84    |  0.75     | 2488   
|     **.+Upper**    |   0.65       | 0.52    |  0.57     |  274   
|     **:**    |   0.52       | 0.31    |  0.39     |   39   
|     **:+Upper**    |   0.36       | 0.62    |  0.45     |   16   
|     **;**    |   0.00       | 0.00    |  0.00     |   17   
|     **?**    |   0.54       | 0.48    |  0.51     |   46   
|     **?+Upper**    |   0.40       | 0.50    |  0.44     |    4   
|     **none**    |   0.96       | 0.96    |  0.96     |35352   
|     **Upper**    |   0.84       | 0.82    |  0.83     | 5442   

-----------------------------------------------
## ☕ Contact 
Contact [Daulet Nurmanbetov](daulet.nurmanbetov@gmail.com) for questions, feedback and/or requests for similar models.

-----------------------------------------------