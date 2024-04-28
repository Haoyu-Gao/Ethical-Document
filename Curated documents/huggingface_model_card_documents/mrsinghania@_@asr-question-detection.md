---
{}
---
<i>Question vs Statement classifier</i> trained on more than 7k samples which were coming from spoken data in an interview setting

<b>Code for using in Transformers:</b>

from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("mrsinghania/asr-question-detection")

model = AutoModelForSequenceClassification.from_pretrained("mrsinghania/asr-question-detection")