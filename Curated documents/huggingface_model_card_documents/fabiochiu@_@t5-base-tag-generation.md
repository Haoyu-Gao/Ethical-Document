---
license: apache-2.0
tags:
- generated_from_trainer
widget:
- text: Python is a high-level, interpreted, general-purpose programming language.
    Its design philosophy emphasizes code readability with the use of significant
    indentation. Python is dynamically-typed and garbage-collected.
  example_title: Programming
model-index:
- name: t5-base-tag-generation
  results: []
---

# Model description

This model is [t5-base](https://huggingface.co/t5-base) fine-tuned on the [190k Medium Articles](https://www.kaggle.com/datasets/fabiochiusano/medium-articles) dataset for predicting article tags using the article textual content as input. While usually formulated as a multi-label classification problem, this model deals with _tag generation_ as a text2text generation task (inspiration from [text2tags](https://huggingface.co/efederici/text2tags)).
# How to use the model
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")

text = """
Python is a high-level, interpreted, general-purpose programming language. Its
design philosophy emphasizes code readability with the use of significant
indentation. Python is dynamically-typed and garbage-collected.
"""

inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10,
                        max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
tags = list(set(decoded_output.strip().split(", ")))

print(tags)
# ['Programming', 'Code', 'Software Development', 'Programming Languages',
#  'Software', 'Developer', 'Python', 'Software Engineering', 'Science',
#  'Engineering', 'Technology', 'Computer Science', 'Coding', 'Digital', 'Tech',
#  'Python Programming']
```

## Data cleaning

The dataset is composed of Medium articles and their tags. However, each Medium article can have at most five tags, therefore the author needs to choose what he/she believes are the best tags (mainly for SEO-related purposes). This means that an article with the "Python" tag may have not the "Programming Languages" tag, even though the first implies the latter.

To clean the dataset accounting for this problem, a hand-made taxonomy of about 1000 tags was built. Using the taxonomy, the tags of each articles have been augmented (e.g. an article with the "Python" tag will have the "Programming Languages" tag as well, as the taxonomy says that "Python" is part of "Programming Languages"). The taxonomy is not public, if you are interested in it please send an email at chiusanofabio94@gmail.com.

## Training and evaluation data

The model has been trained on a single epoch spanning about 50000 articles, evaluating on 1000 random articles not used during training.

## Evaluation results

- eval_loss: 0.8474
- eval_rouge1: 38.6033
- eval_rouge2: 20.5952
- eval_rougeL: 36.4458
- eval_rougeLsum: 36.3202
- eval_gen_len: 15.257 # average number of generated tokens

## Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.19.2
- Pytorch 1.11.0+cu113
- Datasets 2.2.2
- Tokenizers 0.12.1
