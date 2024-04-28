---
language:
- en
license: mit
datasets:
- mshenoda/spam-messages
pipeline_tag: text-classification
widget:
- text: U have a secret admirer. REVEAL who thinks U R So special. Call 09065174042.
    To opt out Reply REVEAL STOP. 1.50 per msg recd.
  example_title: spam example 1
- text: Hey so this sat are we going for the intro pilates only? Or the kickboxing
    too?
  example_title: ham example 1
- text: Great News! Call FREEFONE 08006344447 to claim your guaranteed $1000 CASH
    or $2000 gift. Speak to a live operator NOW!
  example_title: spam example 2
- text: Dude im no longer a pisces. Im an aquarius now.
  example_title: ham example 2
---
# RoBERTa based Spam Message Detection
Spam messages frequently carry malicious links or phishing attempts posing significant threats to both organizations and their users. By choosing our RoBERTa-based spam message detection system, organizations can greatly enhance their security infrastructure. Our system effectively detects and filters out spam messages, adding an extra layer of security that safeguards organizations against potential financial losses, legal consequences, and reputational harm.

## Model Output
- 0 is ham
- 1 is spam

## Dataset

https://huggingface.co/datasets/mshenoda/spam-messages

The dataset is composed of messages labeled by ham or spam, merged from three data sources:

1. SMS Spam Collection https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Telegram Spam Ham https://huggingface.co/datasets/thehamkercat/telegram-spam-ham/tree/main
3. Enron Spam:  https://huggingface.co/datasets/SetFit/enron_spam/tree/main (only used message column and labels)

The prepare script for enron is available at https://github.com/mshenoda/roberta-spam/tree/main/data/enron.
The data is split 80% train 10% validation, and 10% test sets; the scripts used to split and merge of the three data sources are available at: https://github.com/mshenoda/roberta-spam/tree/main/data/utils.


## Architecture
The model is fine tuned RoBERTa 

roberta-base: https://huggingface.co/roberta-base

paper: https://arxiv.org/abs/1907.11692 

## Code

https://github.com/mshenoda/roberta-spam