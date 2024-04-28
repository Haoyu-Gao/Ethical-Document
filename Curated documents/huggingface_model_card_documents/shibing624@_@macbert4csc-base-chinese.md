---
language:
- zh
license: apache-2.0
tags:
- bert
- pytorch
- zh
---

# MacBERT for Chinese Spelling Correction(macbert4csc) Model
中文拼写纠错模型

`macbert4csc-base-chinese` evaluate SIGHAN2015 test data：

- Char Level:     precision:0.9372, recall:0.8640, f1:0.8991
- Sentence Level: precision:0.8264, recall:0.7366, f1:0.7789

由于训练使用的数据使用了SIGHAN2015的训练集（复现paper），在SIGHAN2015的测试集上达到SOTA水平。

模型结构，魔改于softmaskedbert：

![arch](arch1.png)

## Usage

本项目开源在中文文本纠错项目：[pycorrector](https://github.com/shibing624/pycorrector)，可支持macbert4csc模型，通过如下命令调用：

```python
from pycorrector.macbert.macbert_corrector import MacBertCorrector

nlp = MacBertCorrector("shibing624/macbert4csc-base-chinese").macbert_correct

i = nlp('今天新情很好')
print(i)
```

当然，你也可使用官方的huggingface/transformers调用：

*Please use 'Bert' related functions to load this model!*

```python
import operator
import torch
from transformers import BertTokenizer, BertForMaskedLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
model.to(device)

texts = ["今天新情很好", "你找到你最喜欢的工作，我也很高心。"]
with torch.no_grad():
    outputs = model(**tokenizer(texts, padding=True, return_tensors='pt').to(device))

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

result = []
for ids, text in zip(outputs.logits, texts):
    _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_text = _text[:len(text)]
    corrected_text, details = get_errors(corrected_text, text)
    print(text, ' => ', corrected_text, details)
    result.append((corrected_text, details))
print(result)
```

output:
```shell
今天新情很好  =>  今天心情很好 [('新', '心', 2, 3)]
你找到你最喜欢的工作，我也很高心。  =>  你找到你最喜欢的工作，我也很高兴。 [('心', '兴', 15, 16)]
```

模型文件组成：
```
macbert4csc-base-chinese
    ├── config.json
    ├── added_tokens.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

### 训练数据集
#### SIGHAN+Wang271K中文纠错数据集


| 数据集 | 语料 | 下载链接 | 压缩包大小 |
| :------- | :--------- | :---------: | :---------: |
| **`SIGHAN+Wang271K中文纠错数据集`** | SIGHAN+Wang271K(27万条) | [百度网盘（密码01b9）](https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ)| 106M |
| **`原始SIGHAN数据集`** | SIGHAN13 14 15 | [官方csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)| 339K |
| **`原始Wang271K数据集`** | Wang271K | [Automatic-Corpus-Generation dimmywang提供](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)| 93M |


SIGHAN+Wang271K中文纠错数据集，数据格式：
```json
[
    {
        "id": "B2-4029-3",
        "original_text": "晚间会听到嗓音，白天的时候大家都不会太在意，但是在睡觉的时候这嗓音成为大家的恶梦。",
        "wrong_ids": [
            5,
            31
        ],
        "correct_text": "晚间会听到噪音，白天的时候大家都不会太在意，但是在睡觉的时候这噪音成为大家的恶梦。"
    },
]
```

```shell
macbert4csc
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

如果需要训练macbert4csc，请参考[https://github.com/shibing624/pycorrector/tree/master/pycorrector/macbert](https://github.com/shibing624/pycorrector/tree/master/pycorrector/macbert)


### About MacBERT
**MacBERT** is an improved BERT with novel **M**LM **a**s **c**orrection pre-training task, which mitigates the discrepancy of pre-training and fine-tuning.

Here is an example of our pre-training task.

| task  | Example       |
| -------------- | ----------------- |
| **Original Sentence**  | we use a language model to predict the probability of the next word. |
|  **MLM** | we use a language [M] to [M] ##di ##ct the pro [M] ##bility of the next word . |
| **Whole word masking**   | we use a language [M] to [M] [M] [M] the [M] [M] [M] of the next word . |
| **N-gram masking** | we use a [M] [M] to [M] [M] [M] the [M] [M] [M] [M] [M] next word . |
| **MLM as correction** | we use a text system to ca ##lc ##ulate the po ##si ##bility of the next word . |

Except for the new pre-training task, we also incorporate the following techniques.

- Whole Word Masking (WWM)
- N-gram masking
- Sentence-Order Prediction (SOP)

**Note that our MacBERT can be directly replaced with the original BERT as there is no differences in the main neural architecture.**

For more technical details, please check our paper: [Revisiting Pre-trained Models for Chinese Natural Language Processing](https://arxiv.org/abs/2004.13922)


## Citation

```latex
@software{pycorrector,
  author = {Xu Ming},
  title = {pycorrector: Text Error Correction Tool},
  year = {2021},
  url = {https://github.com/shibing624/pycorrector},
}
```
