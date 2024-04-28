---
language:
- en
license: mit
tags:
- reward-model
- reward_model
- RLHF
datasets:
- openai/summarize_from_feedback
- openai/webgpt_comparisons
- Dahoas/instruct-synthetic-prompt-responses
- Anthropic/hh-rlhf
metrics:
- accuracy
---
# Reward model trained from human feedback

Reward model (RM) trained to predict which generated answer is better judged by a human, given a question.

RM are useful in these domain:

- QA model evaluation

- serves as reward score in RLHF 

- detect potential toxic response via ranking

All models are train on these dataset with a same split seed across datasets (if validation split wasn't available)

- [webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)

- [summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)

- [synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise)

- [anthropic_hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)

# How to use

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
inputs = tokenizer(question, answer, return_tensors='pt')
score = rank_model(**inputs).logits[0].cpu().detach()
print(score)
```

**Toxic response detection**

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

question = "I just came out of from jail, any suggestion of my future?"
helpful = "It's great to hear that you have been released from jail."
bad = "Go back to jail you scum"

inputs = tokenizer(question, helpful, return_tensors='pt')
good_score = rank_model(**inputs).logits[0].cpu().detach()

inputs = tokenizer(question, bad, return_tensors='pt')
bad_score = rank_model(**inputs).logits[0].cpu().detach()
print(good_score > bad_score) # tensor([True])
```

# Performance

Validation split accuracy

| Model  | [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)  | [Summary](https://huggingface.co/datasets/openai/summarize_from_feedback)  | [SytheticGPT](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise)  | [Anthropic RLHF]() |
|---|---|---|---|---|
| [electra-large-discriminator](https://huggingface.co/OpenAssistant/reward-model-electra-large-discriminator)  | 59.30  | 68.66  | 99.85  | 54.33 |
| **[deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)** | **61.57**  | 71.47  | 99.88  |  **69.25** |
| [deberta-v3-large](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large) | 61.13  | 72.23  | **99.94**  | 55.62 |
| [deberta-v3-base](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-base)  | 59.07  | 66.84  | 99.85  | 54.51  |
| deberta-v2-xxlarge  | 58.67  | **73.27**  | 99.77  | 66.74  |

Its likely SytheticGPT has somekind of surface pattern on the choosen-rejected pair which makes it trivial to differentiate between better the answer.


# Other

Sincere thanks to [stability.ai](https://stability.ai/) for their unwavering support in terms of A100 computational resources. Their contribution was crucial in ensuring the smooth completion of this research project.
