---
language: en
license: apache-2.0
tags:
- sagemaker
- bart
- summarization
datasets:
- samsum
widget:
- text: "Sugi: I am tired of everything in my life. \nTommy: What? How happy you life\
    \ is! I do envy you.\nSugi: You don't know that I have been over-protected by\
    \ my mother these years. I am really about to leave the family and spread my wings.\n\
    Tommy: Maybe you are right. "
model-index:
- name: bart-large-cnn-samsum
  results:
  - task:
      type: abstractive-text-summarization
      name: Abstractive Text Summarization
    dataset:
      name: 'SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization'
      type: samsum
    metrics:
    - type: rogue-1
      value: 43.2111
      name: Validation ROGUE-1
    - type: rogue-2
      value: 22.3519
      name: Validation ROGUE-2
    - type: rogue-l
      value: 33.315
      name: Validation ROGUE-L
    - type: rogue-1
      value: 41.8283
      name: Test ROGUE-1
    - type: rogue-2
      value: 20.9857
      name: Test ROGUE-2
    - type: rogue-l
      value: 32.3602
      name: Test ROGUE-L
---
## `bart-large-cnn-samsum`
This model was trained using Amazon SageMaker and the new Hugging Face Deep Learning container.
For more information look at:
- [🤗 Transformers Documentation: Amazon SageMaker](https://huggingface.co/transformers/sagemaker.html)
- [Example Notebooks](https://github.com/huggingface/notebooks/tree/master/sagemaker)
- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
- [Python SDK SageMaker documentation for Hugging Face](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
- [Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)
## Hyperparameters
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
## Usage
    from transformers import pipeline
    summarizer = pipeline("summarization", model="slauw87/bart-large-cnn-samsum")
    conversation = '''Sugi: I am tired of everything in my life. 
    Tommy: What? How happy you life is! I do envy you.
    Sugi: You don't know that I have been over-protected by my mother these years. I am really about to leave the family and spread my wings.
    Tommy: Maybe you are right.                                           
    '''
    nlp(conversation)
## Results
| key | value |
| --- | ----- |
| eval_rouge1 | 43.2111 |
| eval_rouge2 | 22.3519 |
| eval_rougeL | 33.3153 |
| eval_rougeLsum | 40.0527 |
| predict_rouge1 | 41.8283 |
| predict_rouge2 | 20.9857 |
| predict_rougeL | 32.3602 |
| predict_rougeLsum | 38.7316 |
