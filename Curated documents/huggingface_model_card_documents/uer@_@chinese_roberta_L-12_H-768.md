---
language: zh
datasets: CLUECorpusSmall
widget:
- text: 北京是[MASK]国的首都。
---


# Chinese RoBERTa Miniatures

## Model description

This is the set of 24 Chinese RoBERTa models pre-trained by [UER-py](https://github.com/dbiir/UER-py/), which is introduced in [this paper](https://arxiv.org/abs/1909.05658). Besides, the models could also be pre-trained by [TencentPretrain](https://github.com/Tencent/TencentPretrain) introduced in [this paper](https://arxiv.org/abs/2212.06385), which inherits UER-py to support models with parameters above one billion, and extends it to a multimodal pre-training framework.

[Turc et al.](https://arxiv.org/abs/1908.08962) have shown that the standard BERT recipe is effective on a wide range of model sizes. Following their paper, we released the 24 Chinese RoBERTa models. In order to facilitate users in reproducing the results, we used a publicly available corpus and provided all training details.

You can download the 24 Chinese RoBERTa miniatures either from the [UER-py Modelzoo page](https://github.com/dbiir/UER-py/wiki/Modelzoo), or via HuggingFace from the links below:

|          |           H=128           |           H=256           |            H=512            |            H=768            |
| -------- | :-----------------------: | :-----------------------: | :-------------------------: | :-------------------------: |
| **L=2**  | [**2/128 (Tiny)**][2_128] |      [2/256][2_256]       |       [2/512][2_512]        |       [2/768][2_768]        |
| **L=4**  |      [4/128][4_128]       | [**4/256 (Mini)**][4_256] | [**4/512 (Small)**][4_512]  |       [4/768][4_768]        |
| **L=6**  |      [6/128][6_128]       |      [6/256][6_256]       |       [6/512][6_512]        |       [6/768][6_768]        |
| **L=8**  |      [8/128][8_128]       |      [8/256][8_256]       | [**8/512 (Medium)**][8_512] |       [8/768][8_768]        |
| **L=10** |     [10/128][10_128]      |     [10/256][10_256]      |      [10/512][10_512]       |      [10/768][10_768]       |
| **L=12** |     [12/128][12_128]      |     [12/256][12_256]      |      [12/512][12_512]       | [**12/768 (Base)**][12_768] |

Here are scores on the devlopment set of six Chinese tasks:

| Model          | Score | book_review | chnsenticorp | lcqmc | tnews(CLUE) | iflytek(CLUE) | ocnli(CLUE) |
| -------------- | :---: | :----: | :----------: | :---: | :---------: | :-----------: | :---------: |
| RoBERTa-Tiny   | 72.3  |  83.4  |     91.4     | 81.8  |    62.0     |     55.0      |    60.3     |
| RoBERTa-Mini   | 75.9  |  85.7  |     93.7     | 86.1  |    63.9     |     58.3      |    67.4     |
| RoBERTa-Small  | 76.9  |  87.5  |     93.4     | 86.5  |    65.1     |     59.4      |    69.7     |
| RoBERTa-Medium | 78.0  |  88.7  |     94.8     | 88.1  |    65.6     |     59.5      |    71.2     |
| RoBERTa-Base   | 79.7  |  90.1  |     95.2     | 89.2  |    67.0     |     60.9      |    75.5     |

For each task, we selected the best fine-tuning hyperparameters from the lists below, and trained with the sequence length of 128:

- epochs: 3, 5, 8
- batch sizes: 32, 64
- learning rates: 3e-5, 1e-4, 3e-4

## How to use

You can use this model directly with a pipeline for masked language modeling (take the case of RoBERTa-Medium):

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='uer/chinese_roberta_L-8_H-512')
>>> unmasker("中国的首都是[MASK]京。")
[
    {'sequence': '[CLS] 中 国 的 首 都 是 北 京 。 [SEP]', 
     'score': 0.8701988458633423, 
     'token': 1266, 
     'token_str': '北'},
    {'sequence': '[CLS] 中 国 的 首 都 是 南 京 。 [SEP]',
     'score': 0.1194809079170227, 
     'token': 1298, 
     'token_str': '南'},
    {'sequence': '[CLS] 中 国 的 首 都 是 东 京 。 [SEP]', 
     'score': 0.0037803512532263994, 
     'token': 691, 
     'token_str': '东'},
    {'sequence': '[CLS] 中 国 的 首 都 是 普 京 。 [SEP]',
     'score': 0.0017127094324678183, 
     'token': 3249,
     'token_str': '普'},
    {'sequence': '[CLS] 中 国 的 首 都 是 望 京 。 [SEP]',
     'score': 0.001687526935711503,
     'token': 3307, 
     'token_str': '望'}
]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-8_H-512')
model = BertModel.from_pretrained("uer/chinese_roberta_L-8_H-512")
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-8_H-512')
model = TFBertModel.from_pretrained("uer/chinese_roberta_L-8_H-512")
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

[CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020/) is used as training data. We found that models pre-trained on CLUECorpusSmall outperform those pre-trained on CLUECorpus2020, although CLUECorpus2020 is much larger than CLUECorpusSmall.

## Training procedure

Models are pre-trained by [UER-py](https://github.com/dbiir/UER-py/) on [Tencent Cloud](https://cloud.tencent.com/). We pre-train 1,000,000 steps with a sequence length of 128 and then pre-train 250,000 additional steps with a sequence length of 512. We use the same hyper-parameters on different model sizes.

Taking the case of RoBERTa-Medium

Stage1:

```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path cluecorpussmall_seq128_dataset.pt \
                      --processes_num 32 --seq_length 128 \
                      --dynamic_masking --data_processor mlm
```

```
python3 pretrain.py --dataset_path cluecorpussmall_seq128_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert/medium_config.json \
                    --output_model_path models/cluecorpussmall_roberta_medium_seq128_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 1000000 --save_checkpoint_steps 100000 --report_steps 50000 \
                    --learning_rate 1e-4 --batch_size 64 \
                    --data_processor mlm --target mlm
```

Stage2:

```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path cluecorpussmall_seq512_dataset.pt \
                      --processes_num 32 --seq_length 512 \
                      --dynamic_masking --data_processor mlm
```

```
python3 pretrain.py --dataset_path cluecorpussmall_seq512_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/cluecorpussmall_roberta_medium_seq128_model.bin-1000000 \
                    --config_path models/bert/medium_config.json \
                    --output_model_path models/cluecorpussmall_roberta_medium_seq512_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 250000 --save_checkpoint_steps 50000 --report_steps 10000 \
                    --learning_rate 5e-5 --batch_size 16 \
                    --data_processor mlm --target mlm
```

Finally, we convert the pre-trained model into Huggingface's format:

```
python3 scripts/convert_bert_from_uer_to_huggingface.py --input_model_path models/cluecorpussmall_roberta_medium_seq512_model.bin-250000 \                                                        
                                                        --output_model_path pytorch_model.bin \
                                                        --layers_num 8 --type mlm
```

### BibTeX entry and citation info

```
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@article{liu2019roberta,
  title={Roberta: A robustly optimized bert pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{turc2019,
  title={Well-Read Students Learn Better: On the Importance of Pre-training Compact Models},
  author={Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1908.08962v2 },
  year={2019}
}

@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}

@article{zhao2023tencentpretrain,
  title={TencentPretrain: A Scalable and Flexible Toolkit for Pre-training Models of Different Modalities},
  author={Zhao, Zhe and Li, Yudong and Hou, Cheng and Zhao, Jing and others},
  journal={ACL 2023},
  pages={217},
  year={2023}
}
```

[2_128]:https://huggingface.co/uer/chinese_roberta_L-2_H-128
[2_256]:https://huggingface.co/uer/chinese_roberta_L-2_H-256
[2_512]:https://huggingface.co/uer/chinese_roberta_L-2_H-512
[2_768]:https://huggingface.co/uer/chinese_roberta_L-2_H-768
[4_128]:https://huggingface.co/uer/chinese_roberta_L-4_H-128
[4_256]:https://huggingface.co/uer/chinese_roberta_L-4_H-256
[4_512]:https://huggingface.co/uer/chinese_roberta_L-4_H-512
[4_768]:https://huggingface.co/uer/chinese_roberta_L-4_H-768
[6_128]:https://huggingface.co/uer/chinese_roberta_L-6_H-128
[6_256]:https://huggingface.co/uer/chinese_roberta_L-6_H-256
[6_512]:https://huggingface.co/uer/chinese_roberta_L-6_H-512
[6_768]:https://huggingface.co/uer/chinese_roberta_L-6_H-768
[8_128]:https://huggingface.co/uer/chinese_roberta_L-8_H-128
[8_256]:https://huggingface.co/uer/chinese_roberta_L-8_H-256
[8_512]:https://huggingface.co/uer/chinese_roberta_L-8_H-512
[8_768]:https://huggingface.co/uer/chinese_roberta_L-8_H-768
[10_128]:https://huggingface.co/uer/chinese_roberta_L-10_H-128
[10_256]:https://huggingface.co/uer/chinese_roberta_L-10_H-256
[10_512]:https://huggingface.co/uer/chinese_roberta_L-10_H-512
[10_768]:https://huggingface.co/uer/chinese_roberta_L-10_H-768
[12_128]:https://huggingface.co/uer/chinese_roberta_L-12_H-128
[12_256]:https://huggingface.co/uer/chinese_roberta_L-12_H-256
[12_512]:https://huggingface.co/uer/chinese_roberta_L-12_H-512
[12_768]:https://huggingface.co/uer/chinese_roberta_L-12_H-768