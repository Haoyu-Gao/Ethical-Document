---
language: en
tags:
- autonlp
datasets:
- madhurjindal/autonlp-data-Gibberish-Detector
widget:
- text: I love Machine Learning!
co2_eq_emissions: 5.527544460835904
---

# Problem Description
The ability to process and understand user input is crucial for various applications, such as chatbots or downstream tasks. However, a common challenge faced in such systems is the presence of gibberish or nonsensical input. To address this problem, we present a project focused on developing a gibberish detector for the English language.
The primary goal of this project is to classify user input as either **gibberish** or **non-gibberish**, enabling more accurate and meaningful interactions with the system. We also aim to enhance the overall performance and user experience of chatbots and other systems that rely on user input.

>## What is Gibberish?
Gibberish refers to **nonsensical or meaningless language or text** that lacks coherence or any discernible meaning. It can be characterized by a combination of random words, nonsensical phrases, grammatical errors, or syntactical abnormalities that prevent the communication from conveying a clear and understandable message. Gibberish can vary in intensity, ranging from simple noise with no meaningful words to sentences that may appear superficially correct but lack coherence or logical structure when examined closely. Detecting and identifying gibberish is essential in various contexts, such as **natural language processing**, **chatbot systems**, **spam filtering**, and **language-based security measures**, to ensure effective communication and accurate processing of user inputs.

## Label Description
Thus, we break down the problem into 4 categories:

1. **Noise:** Gibberish at the zero level where even the different constituents of the input phrase (words) do not hold any meaning independently.  
   *For example: `dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq.`*
   
2. **Word Salad:** Gibberish at level 1 where words make sense independently, but when looked at the bigger picture (the phrase) any meaning is not depicted.  
   *For example: `22 madhur old punjab pickle chennai`*

3. **Mild gibberish:** Gibberish at level 2 where there is a part of the sentence that has grammatical errors, word sense errors, or any syntactical abnormalities, which leads the sentence to miss out on a coherent meaning.  
   *For example: `Madhur study in a teacher`*

4. **Clean:** This category represents a set of words that form a complete and meaningful sentence on its own.  
   *For example: `I love this website`*

> **Tip:** To facilitate gibberish detection, you can combine the labels based on the desired level of detection. For instance, if you need to detect gibberish at level 1, you can group Noise and Word Salad together as "Gibberish," while considering Mild gibberish and Clean separately as "NotGibberish." This approach allows for flexibility in detecting and categorizing different levels of gibberish based on specific requirements.


# Model Trained Using AutoNLP

- Problem type: Multi-class Classification
- Model ID: 492513457
- CO2 Emissions (in grams): 5.527544460835904

## Validation Metrics

- Loss: 0.07609463483095169
- Accuracy: 0.9735624586913417
- Macro F1: 0.9736173135739408
- Micro F1: 0.9735624586913417
- Weighted F1: 0.9736173135739408
- Macro Precision: 0.9737771415197378
- Micro Precision: 0.9735624586913417
- Weighted Precision: 0.9737771415197378
- Macro Recall: 0.9735624586913417
- Micro Recall: 0.9735624586913417
- Weighted Recall: 0.9735624586913417


## Usage

You can use cURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "I love Machine Learning!"}' https://api-inference.huggingface.co/models/madhurjindal/autonlp-Gibberish-Detector-492513457
```

Or Python API:

```
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", use_auth_token=True)

inputs = tokenizer("I love Machine Learning!", return_tensors="pt")

outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=-1)

predicted_index = torch.argmax(probs, dim=1).item()

predicted_prob = probs[0][predicted_index].item()

labels = model.config.id2label

predicted_label = labels[predicted_index]

for i, prob in enumerate(probs[0]):
    print(f"Class: {labels[i]}, Probability: {prob:.4f}")
```

Another simplifed solution with transformers pipline:

```
from transformers import pipeline
selected_model = "madhurjindal/autonlp-Gibberish-Detector-492513457"
classifier = pipeline("text-classification", model=selected_model)
classifier("I love Machine Learning!")
```