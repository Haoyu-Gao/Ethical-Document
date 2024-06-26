---
language:
- ja
license:
- cc-by-4.0
tags:
- NER
- medical documents
datasets:
- MedTxt-CR-JA-training-v2.xml
metrics:
- NTCIR-16 Real-MedNLP subtask 1
---


This is a model for named entity recognition of Japanese medical documents.

### How to use

Download the following five files and put into the same folder.
- id_to_tags.pkl
- key_attr.pkl
- NER_medNLP.py
- predict.py
- text.txt (This is an input file which should be predicted, which could be changed.)

You can use this model by running predict.py.

```
python3 predict.py
```

### Input Example

```
肥大型心筋症、心房細動に対してＷＦ投与が開始となった。
治療経過中に非持続性心室頻拍が認められたためアミオダロンが併用となった。
```

### Output Example

```
 <d certainty="positive">肥大型心筋症、心房細動</d>に対して<m-key state="executed">ＷＦ</m-key>投与が開始となった。
<timex3 type="med">治療経過中</timex3>に<d certainty="positive">非持続性心室頻拍</d>が認められたため<m-key state="executed">アミオダロン</m-key>が併用となった。
```

### Publication

Tomohiro Nishiyama, Aki Ando, Mihiro Nishidani, Shuntaro Yada, Shoko Wakamiya, Eiji Aramaki: NAISTSOC at the NTCIR-16 Real-MedNLP Task, In Proceedings of the 16th NTCIR Conference on Evaluation of Information Access Technologies (NTCIR-16), pp. 330-333, 2022
