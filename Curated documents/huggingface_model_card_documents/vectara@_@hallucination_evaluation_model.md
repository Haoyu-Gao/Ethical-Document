---
language: en
license: apache-2.0
tags:
- microsoft/deberta-v3-base
datasets:
- multi_nli
- snli
- fever
- tals/vitaminc
- paws
metrics:
- accuracy
- auc
- balanced accuracy
pipeline_tag: text-classification
widget:
- text: A man walks into a bar and buys a drink [SEP] A bloke swigs alcohol at a pub
  example_title: Positive
- text: A boy is jumping on skateboard in the middle of a red bridge. [SEP] The boy
    skates down the sidewalk on a blue bridge
  example_title: Negative
---
<img src="candle.png" width="50" height="50" style="display: inline;">  In Loving memory of Simon Mark Hughes...

# Cross-Encoder for Hallucination Detection
This model was trained using [SentenceTransformers](https://sbert.net) [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) class. 
The model outputs a probabilitity from 0 to 1, 0 being a hallucination and 1 being factually consistent. 
The predictions can be thresholded at 0.5 to predict whether a document is consistent with its source.

## Training Data
This model is based on [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) and is trained initially on NLI data to determine textual entailment, before being further fine tuned on summarization datasets with samples annotated for factual consistency including [FEVER](https://huggingface.co/datasets/fever), [Vitamin C](https://huggingface.co/datasets/tals/vitaminc) and [PAWS](https://huggingface.co/datasets/paws).

## Performance

* [TRUE Dataset](https://arxiv.org/pdf/2204.04991.pdf) (Minus Vitamin C, FEVER and PAWS) - 0.872 AUC Score
* [SummaC Benchmark](https://aclanthology.org/2022.tacl-1.10.pdf) (Test Split) - 0.764 Balanced Accuracy, 0.831 AUC Score
* [AnyScale Ranking Test for Hallucinations](https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper) - 86.6 % Accuracy

## LLM Hallucination Leaderboard
If you want to stay up to date with results of the latest tests using this model to evaluate the top LLM models, a public leaderboard is maintained and periodically updated on the [vectara/hallucination-leaderboard](https://github.com/vectara/hallucination-leaderboard) GitHub repository.

## Note about using the Inference API Widget on the Right
To use the model with the widget, you need to pass both documents as a single string separated with [SEP]. For example:

* A man walks into a bar and buys a drink [SEP] A bloke swigs alcohol at a pub
* A person on a horse jumps over a broken down airplane. [SEP] A person is at a diner, ordering an omelette.
* A person on a horse jumps over a broken down airplane. [SEP] A person is outdoors, on a horse.

etc. See examples below for expected probability scores.

## Usage with Sentencer Transformers (Recommended)

### Inference
The model can be used like this, on pairs of documents, passed as a list of list of strings (```List[List[str]]]```):

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('vectara/hallucination_evaluation_model')
scores = model.predict([
    ["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],
    ["A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette."],
    ["A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."],
    ["A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a blue bridge"],
    ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond drinking water in public."],
    ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."],
    ["Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."],  
])
```

This returns a numpy array representing a factual consistency score. A score < 0.5 indicates a likely hallucination):
```
array([0.61051559, 0.00047493709, 0.99639291, 0.00021221573, 0.99599433, 0.0014127002, 0.002.8262993], dtype=float32)
```

Note that the model is designed to work with entire documents, so long as they fit into the 512 token context window (across both documents). 
Also note that the order of the documents is important, the first document is the source document, and the second document is validated against the first for factual consistency, e.g. as a summary of the first or a claim drawn from the source.

### Training

```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample

num_epochs = 5
model_save_path = "./model_dump"
model_name = 'cross-encoder/nli-deberta-v3-base' # base model, use 'vectara/hallucination_evaluation_model' if you want to further fine-tune ours

model = CrossEncoder(model_name, num_labels=1, automodel_args={'ignore_mismatched_sizes':True})

# Load some training examples as such, using a pandas dataframe with source and summary columns:
train_examples, test_examples = [], []
for i, row in df_train.iterrows():
    train_examples.append(InputExample(texts=[row['source'], row['summary']], label=int(row['label'])))

for i, row in df_test.iterrows():
    test_examples.append(InputExample(texts=[row['source'], row['summary']], label=int(row['label'])))
test_evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_examples, name='test_eval')

# Then train the model as such as per the Cross Encoder API:
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
model.fit(train_dataloader=train_dataloader,
          evaluator=test_evaluator,
          epochs=num_epochs,
          evaluation_steps=10_000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          show_progress_bar=True)
```

## Usage with Transformers AutoModel
You can use the model also directly with Transformers library (without the SentenceTransformers library):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model')
tokenizer = AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model')

pairs = [
    ["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],
    ["A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette."],
    ["A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."],
    ["A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a blue bridge"],
    ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond drinking water in public."],
    ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."],
    ["Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."], 
]

inputs = tokenizer.batch_encode_plus(pairs, return_tensors='pt', padding=True)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits.cpu().detach().numpy()
    # convert logits to probabilities
    scores = 1 / (1 + np.exp(-logits)).flatten()
```

This returns a numpy array representing a factual consistency score. A score < 0.5 indicates a likely hallucination):
```
array([0.61051559, 0.00047493709, 0.99639291, 0.00021221573, 0.99599433, 0.0014127002, 0.002.8262993], dtype=float32)
```

## Contact Details
Feel free to contact us on 
* X/Twitter - https://twitter.com/vectara or http://twitter.com/ofermend
* Discussion [forums](https://discuss.vectara.com/)
* Discord [server](https://discord.gg/GFb8gMz6UH)