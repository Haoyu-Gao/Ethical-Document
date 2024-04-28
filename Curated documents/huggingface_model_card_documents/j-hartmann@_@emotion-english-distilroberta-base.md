---
language: en
tags:
- distilroberta
- sentiment
- emotion
- twitter
- reddit
widget:
- text: Oh wow. I didn't know that.
- text: This movie always makes me cry..
- text: Oh Happy Day
---

# Emotion English DistilRoBERTa-base

# Description ℹ

With this model, you can classify emotions in English text data. The model was trained on 6 diverse datasets (see Appendix below) and predicts Ekman's 6 basic emotions, plus a neutral class:

1) anger 🤬
2) disgust 🤢
3) fear 😨
4) joy 😀
5) neutral 😐
6) sadness 😭
7) surprise 😲

The model is a fine-tuned checkpoint of [DistilRoBERTa-base](https://huggingface.co/distilroberta-base). For a 'non-distilled' emotion model, please refer to the model card of the [RoBERTa-large](https://huggingface.co/j-hartmann/emotion-english-roberta-large) version.

# Application 🚀

a) Run emotion model with 3 lines of code on single text example using Hugging Face's pipeline command on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/j-hartmann/emotion-english-distilroberta-base/blob/main/simple_emotion_pipeline.ipynb)

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
classifier("I love this!")
```

```python
Output:
[[{'label': 'anger', 'score': 0.004419783595949411},
  {'label': 'disgust', 'score': 0.0016119900392368436},
  {'label': 'fear', 'score': 0.0004138521908316761},
  {'label': 'joy', 'score': 0.9771687984466553},
  {'label': 'neutral', 'score': 0.005764586851000786},
  {'label': 'sadness', 'score': 0.002092392183840275},
  {'label': 'surprise', 'score': 0.008528684265911579}]]
```

b) Run emotion model on multiple examples and full datasets (e.g., .csv files) on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/j-hartmann/emotion-english-distilroberta-base/blob/main/emotion_prediction_example.ipynb)

# Contact 💻

Please reach out to [jochen.hartmann@tum.de](mailto:jochen.hartmann@tum.de) if you have any questions or feedback.

Thanks to Samuel Domdey and [chrsiebert](https://huggingface.co/siebert) for their support in making this model available.

# Reference ✅

For attribution, please cite the following reference if you use this model. A working paper will be available soon.

```
Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
```

BibTex citation:

```
@misc{hartmann2022emotionenglish,
  author={Hartmann, Jochen},
  title={Emotion English DistilRoBERTa-base},
  year={2022},
  howpublished = {\url{https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/}},
}
```

# Appendix 📚

Please find an overview of the datasets used for training below. All datasets contain English text. The table summarizes which emotions are available in each of the datasets. The datasets represent a diverse collection of text types. Specifically, they contain emotion labels for texts from Twitter, Reddit, student self-reports, and utterances from TV dialogues. As MELD (Multimodal EmotionLines Dataset) extends the popular EmotionLines dataset, EmotionLines itself is not included here. 

|Name|anger|disgust|fear|joy|neutral|sadness|surprise|
|---|---|---|---|---|---|---|---|
|Crowdflower (2016)|Yes|-|-|Yes|Yes|Yes|Yes|
|Emotion Dataset, Elvis et al. (2018)|Yes|-|Yes|Yes|-|Yes|Yes|
|GoEmotions, Demszky et al. (2020)|Yes|Yes|Yes|Yes|Yes|Yes|Yes|
|ISEAR, Vikash (2018)|Yes|Yes|Yes|Yes|-|Yes|-|
|MELD, Poria et al. (2019)|Yes|Yes|Yes|Yes|Yes|Yes|Yes|
|SemEval-2018, EI-reg, Mohammad et al. (2018) |Yes|-|Yes|Yes|-|Yes|-|

The model is trained on a balanced subset from the datasets listed above (2,811 observations per emotion, i.e., nearly 20k observations in total). 80% of this balanced subset is used for training and 20% for evaluation. The evaluation accuracy is 66% (vs. the random-chance baseline of 1/7 = 14%).

# Scientific Applications 📖

Below you can find a list of papers using "Emotion English DistilRoBERTa-base". If you would like your paper to be added to the list, please send me an email.

- Butt, S., Sharma, S., Sharma, R., Sidorov, G., & Gelbukh, A. (2022). What goes on inside rumour and non-rumour tweets and their reactions: A Psycholinguistic Analyses. Computers in Human Behavior, 107345.
- Kuang, Z., Zong, S., Zhang, J., Chen, J., & Liu, H. (2022). Music-to-Text Synaesthesia: Generating Descriptive Text from Music Recordings. arXiv preprint arXiv:2210.00434.
- Rozado, D., Hughes, R., & Halberstadt, J. (2022). Longitudinal analysis of sentiment and emotion in news media headlines using automated labelling with Transformer language models. Plos one, 17(10), e0276367.