---
language: en
license: mit
tags:
- sagemaker
- bart
- summarization
datasets:
- samsum
widget:
- text: "Jeff: Can I train a \U0001F917 Transformers model on Amazon SageMaker? \n\
    Philipp: Sure you can use the new Hugging Face Deep Learning Container. \nJeff:\
    \ ok.\nJeff: and how can I get started? \nJeff: where can I find documentation?\
    \ \nPhilipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face\n"
model-index:
- name: bart-large-cnn-samsum
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: 'SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization'
      type: samsum
    metrics:
    - type: rogue-1
      value: 42.621
      name: Validation ROGUE-1
    - type: rogue-2
      value: 21.9825
      name: Validation ROGUE-2
    - type: rogue-l
      value: 33.034
      name: Validation ROGUE-L
    - type: rogue-1
      value: 41.3174
      name: Test ROGUE-1
    - type: rogue-2
      value: 20.8716
      name: Test ROGUE-2
    - type: rogue-l
      value: 32.1337
      name: Test ROGUE-L
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
    metrics:
    - type: rouge
      value: 41.3282
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTYzNzZkZDUzOWQzNGYxYTJhNGE4YWYyZjA0NzMyOWUzMDNhMmVhYzY1YTM0ZTJhYjliNGE4MDZhMjhhYjRkYSIsInZlcnNpb24iOjF9.OOM6l3v5rJCndmUIJV-2SDh2NjbPo5IgQOSL-Ju1Gwbi1voL5amsDEDOelaqlUBE3n55KkUsMLZhyn66yWxZBQ
    - type: rouge
      value: 20.8755
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWZiODFiYWQzY2NmOTc5YjA3NTI0YzQ1MzQ0ODk2NjgyMmVlMjA5MjZiNTJkMGRmZGEzN2M3MDNkMjkxMDVhYSIsInZlcnNpb24iOjF9.b8cPk2-IL24La3Vd0hhtii4tRXujh5urAwy6IVeTWHwYfXaURyC2CcQOWtlOx5bdO5KACeaJFrFBCGgjk-VGCQ
    - type: rouge
      value: 32.1353
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYWNmYzdiYWQ2ZWRkYzRiMGMxNWUwODgwZTdkY2NjZTc1NWE5NTFiMzU0OTU1N2JjN2ExYWQ2NGZkNjk5OTc4YSIsInZlcnNpb24iOjF9.Fzv4p-TEVicljiCqsBJHK1GsnE_AwGqamVmxTPI0WBNSIhZEhliRGmIL_z1pDq6WOzv3GN2YUGvhowU7GxnyAQ
    - type: rouge
      value: 38.401
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNGI4MWY0NWMxMmQ0ODQ5MDhiNDczMDAzYzJkODBiMzgzYWNkMWM2YTZkZDJmNWJiOGQ3MmNjMGViN2UzYWI2ZSIsInZlcnNpb24iOjF9.7lw3h5k5lJ7tYFLZGUtLyDabFYd00l6ByhmvkW4fykocBy9Blyin4tdw4Xps4DW-pmrdMLgidHxBWz5MrSx1Bw
    - type: loss
      value: 1.4297215938568115
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzI0ZWNhNDM5YTViZDMyZGJjMDA1ZWFjYzNhOTdlOTFiNzhhMDBjNmM2MjA3ZmRkZjJjMjEyMGY3MzcwOTI2NyIsInZlcnNpb24iOjF9.oNaZsAtUDqGAqoZWJavlcW7PKx1AWsnkbhaQxadpOKk_u7ywJJabvTtzyx_DwEgZslgDETCf4MM-JKitZKjiDA
    - type: gen_len
      value: 60.0757
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYTgwYWYwMDRkNTJkMDM5N2I2MWNmYzQ3OWM1NDJmODUyZGViMGE4ZTdkNmIwYWM2N2VjZDNmN2RiMDE4YTYyYiIsInZlcnNpb24iOjF9.PbXTcNYX_SW-BuRQEcqyc21M7uKrOMbffQSAK6k2GLzTVRrzZxsDC57ktKL68zRY8fSiRGsnknOwv-nAR6YBCQ
---

## `bart-large-cnn-samsum`

> If you want to use the model you should try a newer fine-tuned FLAN-T5 version [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum) out socring the BART version with `+6` on `ROGUE1` achieving `47.24`.

# TRY [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum)


This model was trained using Amazon SageMaker and the new Hugging Face Deep Learning container.

For more information look at:
- [🤗 Transformers Documentation: Amazon SageMaker](https://huggingface.co/transformers/sagemaker.html)
- [Example Notebooks](https://github.com/huggingface/notebooks/tree/master/sagemaker)
- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
- [Python SDK SageMaker documentation for Hugging Face](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
- [Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)

## Hyperparameters
```json
{
    "dataset_name": "samsum",
    "do_eval": true,
    "do_predict": true,
    "do_train": true,
    "fp16": true,
    "learning_rate": 5e-05,
    "model_name_or_path": "facebook/bart-large-cnn",
    "num_train_epochs": 3,
    "output_dir": "/opt/ml/model",
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "predict_with_generate": true,
    "seed": 7
}
```

## Usage
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
'''
summarizer(conversation)
```

## Results

| key | value |
| --- | ----- |
| eval_rouge1 | 42.621 |
| eval_rouge2 | 21.9825 |
| eval_rougeL | 33.034 |
| eval_rougeLsum | 39.6783 |
| test_rouge1 | 41.3174 |
| test_rouge2 | 20.8716 |
| test_rougeL | 32.1337 |
| test_rougeLsum | 38.4149 |

