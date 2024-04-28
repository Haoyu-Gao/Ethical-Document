---
language: en
license: mit
tags:
- keyphrase-extraction
datasets:
- midas/inspec
metrics:
- seqeval
widget:
- text: 'Keyphrase extraction is a technique in text analysis where you extract the
    important keyphrases from a document. Thanks to these keyphrases humans can understand
    the content of a text very quickly and easily without reading it completely. Keyphrase
    extraction was first done primarily by human annotators, who read the text in
    detail and then wrote down the most important keyphrases. The disadvantage is
    that if you work with a lot of documents, this process can take a lot of time.

    Here is where Artificial Intelligence comes in. Currently, classical machine learning
    methods, that use statistical and linguistic features, are widely used for the
    extraction process. Now with deep learning, it is possible to capture the semantic
    meaning of a text even better than these classical methods. Classical methods
    look at the frequency, occurrence and order of words in the text, whereas these
    neural approaches can capture long-term semantic dependencies and context of words
    in a text.'
  example_title: Example 1
- text: In this work, we explore how to learn task specific language models aimed
    towards learning rich representation of keyphrases from text documents. We experiment
    with different masking strategies for pre-training transformer language models
    (LMs) in discriminative as well as generative settings. In the discriminative
    setting, we introduce a new pre-training objective - Keyphrase Boundary Infilling
    with Replacement (KBIR), showing large gains in performance (up to 9.26 points
    in F1) over SOTA, when LM pre-trained using KBIR is fine-tuned for the task of
    keyphrase extraction. In the generative setting, we introduce a new pre-training
    setup for BART - KeyBART, that reproduces the keyphrases related to the input
    text in the CatSeq format, instead of the denoised original input. This also led
    to gains in performance (up to 4.33 points inF1@M) over SOTA for keyphrase generation.
    Additionally, we also fine-tune the pre-trained language models on named entity
    recognition(NER), question answering (QA), relation extraction (RE), abstractive
    summarization and achieve comparable performance with that of the SOTA, showing
    that learning rich representation of keyphrases is indeed beneficial for many
    other fundamental NLP tasks.
  example_title: Example 2
model-index:
- name: DeDeckerThomas/keyphrase-extraction-distilbert-inspec
  results:
  - task:
      type: keyphrase-extraction
      name: Keyphrase Extraction
    dataset:
      name: inspec
      type: midas/inspec
    metrics:
    - type: F1 (Seqeval)
      value: 0.509
      name: F1 (Seqeval)
    - type: F1@M
      value: 0.49
      name: F1@M
---
# 🔑 Keyphrase Extraction Model: distilbert-inspec
Keyphrase extraction is a technique in text analysis where you extract the important keyphrases from a document. Thanks to these keyphrases humans can understand the content of a text very quickly and easily without reading it completely. Keyphrase extraction was first done primarily by human annotators, who read the text in detail and then wrote down the most important keyphrases. The disadvantage is that if you work with a lot of documents, this process can take a lot of time ⏳. 

Here is where Artificial Intelligence 🤖 comes in. Currently, classical machine learning methods, that use statistical and linguistic features, are widely used for the extraction process. Now with deep learning, it is possible to capture the semantic meaning of a text even better than these classical methods. Classical methods look at the frequency, occurrence and order of words in the text, whereas these neural approaches can capture long-term semantic dependencies and context of words in a text.


## 📓 Model Description
This model uses [distilbert](https://huggingface.co/distilbert-base-uncased) as its base model and fine-tunes it on the [Inspec dataset](https://huggingface.co/datasets/midas/inspec).

Keyphrase extraction models are transformer models fine-tuned as a token classification problem where each word in the document is classified as being part of a keyphrase or not.

| Label | Description                     |
| ----- | ------------------------------- |
| B-KEY | At the beginning of a keyphrase |
| I-KEY | Inside a keyphrase              |
| O     | Outside a keyphrase             |

Kulkarni, Mayank, Debanjan Mahata, Ravneet Arora, and Rajarshi Bhowmik. "Learning Rich Representation of Keyphrases from Text." arXiv preprint arXiv:2112.08547 (2021).

Sahrawat, Dhruva, Debanjan Mahata, Haimin Zhang, Mayank Kulkarni, Agniv Sharma, Rakesh Gosangi, Amanda Stent, Yaman Kumar, Rajiv Ratn Shah, and Roger Zimmermann. "Keyphrase extraction as sequence labeling using contextualized embeddings." In European Conference on Information Retrieval, pp. 328-335. Springer, Cham, 2020.

## ✋ Intended Uses & Limitations
### 🛑 Limitations
* This keyphrase extraction model is very domain-specific and will perform very well on abstracts of scientific papers. It's not recommended to use this model for other domains, but you are free to test it out.
* Only works for English documents.

### ❓ How To Use
```python
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])

```

```python
# Load pipeline
model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)
```
```python
# Inference
text = """
Keyphrase extraction is a technique in text analysis where you extract the
important keyphrases from a document. Thanks to these keyphrases humans can
understand the content of a text very quickly and easily without reading it
completely. Keyphrase extraction was first done primarily by human annotators,
who read the text in detail and then wrote down the most important keyphrases.
The disadvantage is that if you work with a lot of documents, this process
can take a lot of time. 

Here is where Artificial Intelligence comes in. Currently, classical machine
learning methods, that use statistical and linguistic features, are widely used
for the extraction process. Now with deep learning, it is possible to capture
the semantic meaning of a text even better than these classical methods.
Classical methods look at the frequency, occurrence and order of words
in the text, whereas these neural approaches can capture long-term
semantic dependencies and context of words in a text.
""".replace("\n", " ")

keyphrases = extractor(text)

print(keyphrases)
```

```
# Output
['artificial intelligence' 'classical machine learning' 'deep learning'
 'keyphrase extraction' 'linguistic features' 'statistical'
 'text analysis']
```

## 📚 Training Dataset
[Inspec](https://huggingface.co/datasets/midas/inspec) is a keyphrase extraction/generation dataset consisting of 2000 English scientific papers from the scientific domains of Computers and Control and Information Technology published between 1998 to 2002. The keyphrases are annotated by professional indexers or editors.

You can find more information in the [paper](https://dl.acm.org/doi/10.3115/1119355.1119383).

## 👷‍♂️ Training Procedure
### Training Parameters

| Parameter | Value |
| --------- | ------|
| Learning Rate | 1e-4 |
| Epochs | 50 |
| Early Stopping Patience | 3 |

### Preprocessing
The documents in the dataset are already preprocessed into list of words with the corresponding labels. The only thing that must be done is tokenization and the realignment of the labels so that they correspond with the right subword tokens.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Labels
label_list = ["B", "I", "O"]
lbl2idx = {"B": 0, "I": 1, "O": 2}
idx2label = {0: "B", 1: "I", 2: "O"}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
max_length = 512

# Dataset parameters
dataset_full_name = "midas/inspec"
dataset_subset = "raw"
dataset_document_column = "document"
dataset_biotags_column = "doc_bio_tags"

def preprocess_fuction(all_samples_per_split):
    tokenized_samples = tokenizer.batch_encode_plus(
        all_samples_per_split[dataset_document_column],
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    total_adjusted_labels = []
    for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = all_samples_per_split[dataset_biotags_column][k]
        i = -1
        adjusted_label_ids = []

        for wid in word_ids_list:
            if wid is None:
                adjusted_label_ids.append(lbl2idx["O"])
            elif wid != prev_wid:
                i = i + 1
                adjusted_label_ids.append(lbl2idx[existing_label_ids[i]])
                prev_wid = wid
            else:
                adjusted_label_ids.append(
                    lbl2idx[
                        f"{'I' if existing_label_ids[i] == 'B' else existing_label_ids[i]}"
                    ]
                )

        total_adjusted_labels.append(adjusted_label_ids)
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples

# Load dataset
dataset = load_dataset(dataset_full_name, dataset_subset)

# Preprocess dataset
tokenized_dataset = dataset.map(preprocess_fuction, batched=True)
    
```

### Postprocessing (Without Pipeline Function)
If you do not use the pipeline function, you must filter out the B and I labeled tokens. Each B and I will then be merged into a keyphrase. Finally, you need to strip the keyphrases to make sure all unnecessary spaces have been removed.
```python
# Define post_process functions
def concat_tokens_by_tag(keyphrases):
    keyphrase_tokens = []
    for id, label in keyphrases:
        if label == "B":
            keyphrase_tokens.append([id])
        elif label == "I":
            if len(keyphrase_tokens) > 0:
                keyphrase_tokens[len(keyphrase_tokens) - 1].append(id)
    return keyphrase_tokens


def extract_keyphrases(example, predictions, tokenizer, index=0):
    keyphrases_list = [
        (id, idx2label[label])
        for id, label in zip(
            np.array(example["input_ids"]).squeeze().tolist(), predictions[index]
        )
        if idx2label[label] in ["B", "I"]
    ]

    processed_keyphrases = concat_tokens_by_tag(keyphrases_list)
    extracted_kps = tokenizer.batch_decode(
        processed_keyphrases,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return np.unique([kp.strip() for kp in extracted_kps])

```

## 📝 Evaluation Results

Traditional evaluation methods are the precision, recall and F1-score @k,m where k is the number that stands for the first k predicted keyphrases and m for the average amount of predicted keyphrases.
The model achieves the following results on the Inspec test set:

| Dataset           | P@5  | R@5  | F1@5 | P@10 | R@10 | F1@10 | P@M  | R@M  | F1@M |
|:-----------------:|:----:|:----:|:----:|:----:|:----:|:-----:|:----:|:----:|:----:|
| Inspec Test Set   | 0.45 | 0.40 | 0.39 | 0.33 | 0.53 | 0.38  | 0.47 | 0.57 | 0.49 |

## 🚨 Issues
Please feel free to start discussions in the Community Tab.