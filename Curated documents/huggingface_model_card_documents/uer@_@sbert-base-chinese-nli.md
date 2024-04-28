---
language: zh
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
widget:
- source_sentence: 那个人很开心
  sentences:
  - 那个人非常开心
  - 那只猫很开心
  - 那个人在吃东西
---

# Chinese Sentence BERT

## Model description

This is the sentence embedding model pre-trained by [UER-py](https://github.com/dbiir/UER-py/), which is introduced in [this paper](https://arxiv.org/abs/1909.05658). Besides, the model could also be pre-trained by [TencentPretrain](https://github.com/Tencent/TencentPretrain) introduced in [this paper](https://arxiv.org/abs/2212.06385), which inherits UER-py to support models with parameters above one billion, and extends it to a multimodal pre-training framework.

## How to use

You can use this model to extract sentence embeddings for sentence similarity task. We use cosine distance to calculate the embedding similarity here: 

```python
>>> from sentence_transformers import SentenceTransformer
>>> model = SentenceTransformer('uer/sbert-base-chinese-nli')
>>> sentences = ['那个人很开心', '那个人非常开心']
>>> sentence_embeddings = model.encode(sentences)
>>> from sklearn.metrics.pairwise import paired_cosine_distances
>>> cosine_score = 1 - paired_cosine_distances([sentence_embeddings[0]],[sentence_embeddings[1]])
```

## Training data

[ChineseTextualInference](https://github.com/liuhuanyong/ChineseTextualInference/) is used as training data. 

## Training procedure

The model is fine-tuned by [UER-py](https://github.com/dbiir/UER-py/) on [Tencent Cloud](https://cloud.tencent.com/). We fine-tune five epochs with a sequence length of 128 on the basis of the pre-trained model [chinese_roberta_L-12_H-768](https://huggingface.co/uer/chinese_roberta_L-12_H-768). At the end of each epoch, the model is saved when the best performance on development set is achieved.

```
python3 finetune/run_classifier_siamese.py --pretrained_model_path models/cluecorpussmall_roberta_base_seq512_model.bin-250000 \
                                           --vocab_path models/google_zh_vocab.txt \
                                           --config_path models/sbert/base_config.json \
                                           --train_path datasets/ChineseTextualInference/train.tsv \
                                           --dev_path datasets/ChineseTextualInference/dev.tsv \
                                           --learning_rate 5e-5 --epochs_num 5 --batch_size 64
```

Finally, we convert the pre-trained model into Huggingface's format:

```
python3 scripts/convert_sbert_from_uer_to_huggingface.py --input_model_path models/finetuned_model.bin \                                                                
                                                         --output_model_path pytorch_model.bin \                                                                                            
                                                         --layers_num 12
```

### BibTeX entry and citation info

```
@article{reimers2019sentence,
  title={Sentence-bert: Sentence embeddings using siamese bert-networks},
  author={Reimers, Nils and Gurevych, Iryna},
  journal={arXiv preprint arXiv:1908.10084},
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
```