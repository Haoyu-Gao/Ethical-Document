---
language: ko
license: cc-by-4.0
---

# pko-t5-large
[Source Code](https://github.com/paust-team/pko-t5)

pko-t5 는 한국어 전용 데이터로 학습한 [t5 v1.1 모델](https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/released_checkpoints.md)입니다.

한국어를 tokenize 하기 위해서 sentencepiece 대신 OOV 가 없는 BBPE 를 사용했으며 한국어 데이터 (나무위키, 위키피디아, 모두의말뭉치 등..) 를 T5 의 span corruption task 를 사용해서 unsupervised learning 만 적용하여 학습을 진행했습니다.

pko-t5 를 사용하실 때는 대상 task 에 파인튜닝하여 사용하시기 바랍니다.

## Usage
transformers 의 API 를 사용하여 접근 가능합니다. tokenizer 를 사용할때는 `T5Tokenizer` 가 아니라 `T5TokenizerFast` 를 사용해주십시오. model 은 T5ForConditionalGeneration 를 그대로 활용하시면 됩니다.

### Example
```python
from transformers import T5TokenizerFast, T5ForConditionalGeneration

tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-large')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-t5-large')

input_ids = tokenizer(["qa question: 당신의 이름은 무엇인가요?"]).input_ids
labels = tokenizer(["T5 입니다."]).input_ids
outputs = model(input_ids=input_ids, labels=labels)

print(f"loss={outputs.loss} logits={outputs.logits}")
```
    

## Klue 평가 (dev)


|     | Model                                                            | ynat (macro F1) | sts (pearsonr/F1) | nli (acc) | ner (entity-level F1) | re (micro F1) | dp (LAS)  | mrc (EM/F1) |
|-----|------------------------------------------------------------------|-----------------|-------------------|-----------|-----------------------|---------------|-----------|-------------|
|     | Baseline                                                         | **87.30**       | **93.20/86.13**   | **89.50** | 86.06                 | 71.06         | 87.93     | **75.26/-** |
| FT  | [pko-t5-small](https://huggingface.co/paust/pko-t5-small) (77M)  | 86.21           | 77.99/77.01       | 69.20     | 82.60                 | 66.46         | 93.15     | 43.81/46.58 |
| FT  | [pko-t5-base](https://huggingface.co/paust/pko-t5-base) (250M)   | 87.29           | 90.25/83.43       | 79.73     | 87.80                 | 67.23         | 97.28     | 61.53/64.74 |
| FT  | [pko-t5-large](https://huggingface.co/paust/pko-t5-large) (800M) | 87.12           | 92.05/85.24       | 84.96     | **88.18**             | **75.17**     | **97.60** | 68.01/71.44 |
| MT  | pko-t5-small                                                     | 84.54           | 68.50/72/02       | 51.16     | 74.69                 | 66.11         | 80.40     | 43.60/46.28 |
| MT  | pko-t5-base                                                      | 86.89           | 83.96/80.30       | 72.03     | 85.27                 | 66.59         | 95.05     | 61.11/63.94 |
| MT  | pko-t5-large                                                     | 87.57           | 91.93/86.29       | 83.63     | 87.41                 | 71.34         | 96.99     | 70.70/73.72 |

- FT: 싱글태스크 파인튜닝 / MT: 멀티태스크 파인튜닝
- [Baseline](https://arxiv.org/abs/2105.09680): KLUE 논문에서 소개된 dev set 에 대한 SOTA 점수

## License
[PAUST](https://paust.io)에서 만든 pko-t5는 [MIT license](https://github.com/paust-team/pko-t5/blob/main/LICENSE) 하에 공개되어 있습니다.