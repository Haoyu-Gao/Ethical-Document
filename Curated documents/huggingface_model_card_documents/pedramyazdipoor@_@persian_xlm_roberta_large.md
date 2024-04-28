---
{}
---
## Persian XLM-RoBERTA Large For Question Answering Task

XLM-RoBERTA is a multilingual language model pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages. It was introduced in the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116v2) by Conneau et al. .

Multilingual [XLM-RoBERTa large for QA on various languages](https://huggingface.co/deepset/xlm-roberta-large-squad2) is fine-tuned on various QA datasets but PQuAD, which is the biggest persian QA dataset so far. This second model is our base model to be fine-tuned.

Paper presenting PQuAD dataset: [arXiv:2202.06219](https://arxiv.org/abs/2202.06219)

---

## Introduction

This model is fine-tuned on PQuAD Train set and is easily ready to use.
Its very long training time encouraged me to publish this model in order to make life easier for those who need.


## Hyperparameters of training
I set batch size to 4 due to the limitations of GPU memory in Google Colab.
```
batch_size = 4
n_epochs = 1
base_LM_model = "deepset/xlm-roberta-large-squad2"
max_seq_len = 256
learning_rate = 3e-5
evaluation_strategy = "epoch",
save_strategy = "epoch",
learning_rate = 3e-5,
warmup_ratio = 0.1,
gradient_accumulation_steps = 8,
weight_decay = 0.01,
``` 
## Performance
Evaluated on the PQuAD Persian test set with the [official PQuAD link](https://huggingface.co/datasets/newsha/PQuAD).
I trained for more than 1 epoch as well, but I get worse results.
 Our XLM-Roberta outperforms [our ParsBert on PQuAD](https://huggingface.co/pedramyazdipoor/parsbert_question_answering_PQuAD), but the former is more than 3 times bigger than the latter one; so comparing these two is not fair.
### Question Answering On Test Set of PQuAD Dataset

|      Metric      | Our XLM-Roberta Large| Our ParsBert  |
|:----------------:|:--------------------:|:-------------:|
| Exact Match      |   66.56*             | 47.44         |
|      F1          |   87.31*             | 81.96         |



## How to use

## Pytorch
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
path = 'pedramyazdipoor/persian_xlm_roberta_large'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForQuestionAnswering.from_pretrained(path)
```
## Inference 
There are some considerations for inference:
1) Start index of answer must be smaller than end index.
2) The span of answer must be within the context.
3) The selected span must be the most probable choice among N pairs of candidates.

```python
def generate_indexes(start_logits, end_logits, N, min_index):
  
  output_start = start_logits
  output_end = end_logits

  start_indexes = np.arange(len(start_logits))
  start_probs = output_start
  list_start = dict(zip(start_indexes, start_probs.tolist()))
  end_indexes = np.arange(len(end_logits))
  end_probs = output_end
  list_end = dict(zip(end_indexes, end_probs.tolist()))

  sorted_start_list = sorted(list_start.items(), key=lambda x: x[1], reverse=True) #Descending sort by probability
  sorted_end_list = sorted(list_end.items(), key=lambda x: x[1], reverse=True)

  final_start_idx, final_end_idx = [[] for l in range(2)]

  start_idx, end_idx, prob = 0, 0, (start_probs.tolist()[0] + end_probs.tolist()[0])
  for a in range(0,N):
    for b in range(0,N):
      if (sorted_start_list[a][1] + sorted_end_list[b][1]) > prob :
        if (sorted_start_list[a][0] <= sorted_end_list[b][0]) and (sorted_start_list[a][0] > min_index) :
          prob = sorted_start_list[a][1] + sorted_end_list[b][1]
          start_idx = sorted_start_list[a][0]
          end_idx = sorted_end_list[b][0]
  final_start_idx.append(start_idx)    
  final_end_idx.append(end_idx)      

  return final_start_idx[0], final_end_idx[0]
```

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval().to(device)
text = 'سلام من پدرامم 26 سالمه'
question = 'چند سالمه؟'
encoding = tokenizer(question,text,add_special_tokens = True,
                     return_token_type_ids = True,
                     return_tensors = 'pt',
                     padding = True,
                     return_offsets_mapping = True,
                     truncation = 'only_first',
                     max_length = 32)
out = model(encoding['input_ids'].to(device),encoding['attention_mask'].to(device), encoding['token_type_ids'].to(device))
#we had to change some pieces of code to make it compatible with one answer generation at a time
#If you have unanswerable questions, use out['start_logits'][0][0:] and out['end_logits'][0][0:] because <s> (the 1st token) is for this situation and must be compared with other tokens.
#you can initialize min_index in generate_indexes() to put force on tokens being chosen to be within the context(startindex must be greater than seperator token).
answer_start_index, answer_end_index = generate_indexes(out['start_logits'][0][1:], out['end_logits'][0][1:], 5, 0)
print(tokenizer.tokenize(text + question))
print(tokenizer.tokenize(text + question)[answer_start_index : (answer_end_index + 1)])
>>> ['▁سلام', '▁من', '▁پدر', 'ام', 'م', '▁26', '▁سالم', 'ه', 'نام', 'م', '▁چیست', '؟']
>>> ['▁26']
```

## Acknowledgments
We hereby, express our gratitude to the [Newsha Shahbodaghkhan](https://huggingface.co/datasets/newsha/PQuAD/tree/main) for facilitating dataset gathering.
## Contributors
- Pedram Yazdipoor : [Linkedin](https://www.linkedin.com/in/pedram-yazdipour/)
## Releases
### Release v0.2 (Sep 18, 2022)
This is the second version of our Persian XLM-Roberta-Large.
There were some problems using the previous version.