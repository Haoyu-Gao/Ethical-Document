---
language:
- en
license: openrail
library_name: transformers
datasets:
- humarin/chatgpt-paraphrases
inference:
  parameters:
    num_beams: 5
    num_beam_groups: 5
    num_return_sequences: 5
    repetition_penalty: 10.01
    diversity_penalty: 3.01
    no_repeat_ngram_size: 2
    temperature: 0.7
    max_length: 128
widget:
- text: What are the best places to see in New York?
  example_title: New York tourist attractions
- text: When should I go to the doctor?
  example_title: Doctor's time
- text: Rammstein's album Mutter was recorded in the south of France in May and June
    2000, and mixed in Stockholm in October of that year.
  example_title: Rammstein's album Mutter
pipeline_tag: text2text-generation
---
This model was trained on our [ChatGPT paraphrase dataset](https://huggingface.co/datasets/humarin/chatgpt-paraphrases).



This dataset is based on the [Quora paraphrase question](https://www.kaggle.com/competitions/quora-question-pairs), texts from the [SQUAD 2.0](https://huggingface.co/datasets/squad_v2) and the [CNN news dataset](https://huggingface.co/datasets/cnn_dailymail).

This model is based on the T5-base model. We used "transfer learning" to get our model to generate paraphrases as well as ChatGPT. Now we can say that this is one of the best paraphrases of the Hugging Face.

[Kaggle](https://www.kaggle.com/datasets/vladimirvorobevv/chatgpt-paraphrases) link

[Author's LinkedIn](https://www.linkedin.com/in/vladimir-vorobev/) link

## Deploying example
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res
```

## Usage examples

**Input:**
```python
text = 'What are the best places to see in New York?'
paraphrase(text)
```
**Output:**
```python
['What are some must-see places in New York?',
 'Can you suggest some must-see spots in New York?',
 'Where should one go to experience the best NYC has to offer?',
 'Which places should I visit in New York?',
 'What are the top destinations to explore in New York?']
```

**Input:**
```python
text = "Rammstein's album Mutter was recorded in the south of France in May and June 2000, and mixed in Stockholm in October of that year."
paraphrase(text)
```
**Output:**
```python
['In May and June 2000, Rammstein travelled to the south of France to record his album Mutter, which was mixed in Stockholm in October of that year.',
 'The album Mutter by Rammstein was recorded in the south of France during May and June 2000, with mixing taking place in Stockholm in October of that year.',
 'The album Mutter by Rammstein was recorded in the south of France during May and June 2000, with mixing taking place in Stockholm in October of that year. It',
 'Mutter, the album released by Rammstein, was recorded in southern France during May and June 2000, with mixing taking place between October and September.',
 'In May and June 2000, Rammstein recorded his album Mutter in the south of France, with the mix being made at Stockholm during October.']
```


## Train parameters
```python
epochs = 5
batch_size = 64
max_length = 128
lr = 5e-5
batches_qty = 196465
betas = (0.9, 0.999)
eps = 1e-08
```

### BibTeX entry and citation info

```bibtex
@inproceedings{chatgpt_paraphraser,
  author={Vladimir Vorobev, Maxim Kuznetsov},
  title={A paraphrasing model based on ChatGPT paraphrases},
  year={2023}
}
```