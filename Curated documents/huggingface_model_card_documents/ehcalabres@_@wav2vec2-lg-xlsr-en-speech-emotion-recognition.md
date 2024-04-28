---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model_index:
  name: wav2vec2-lg-xlsr-en-speech-emotion-recognition
---

# Speech Emotion Recognition By Fine-Tuning Wav2Vec 2.0

The model is a fine-tuned version of [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) for a Speech Emotion Recognition (SER) task.

The dataset used to fine-tune the original pre-trained model is the [RAVDESS dataset](https://zenodo.org/record/1188976#.YO6yI-gzaUk). This dataset provides 1440 samples of recordings from actors performing on 8 different emotions in English, which are:

```python
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
```

It achieves the following results on the evaluation set:
- Loss: 0.5023
- Accuracy: 0.8223

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 2.0752        | 0.21  | 30   | 2.0505          | 0.1359   |
| 2.0119        | 0.42  | 60   | 1.9340          | 0.2474   |
| 1.8073        | 0.63  | 90   | 1.5169          | 0.3902   |
| 1.5418        | 0.84  | 120  | 1.2373          | 0.5610   |
| 1.1432        | 1.05  | 150  | 1.1579          | 0.5610   |
| 0.9645        | 1.26  | 180  | 0.9610          | 0.6167   |
| 0.8811        | 1.47  | 210  | 0.8063          | 0.7178   |
| 0.8756        | 1.68  | 240  | 0.7379          | 0.7352   |
| 0.8208        | 1.89  | 270  | 0.6839          | 0.7596   |
| 0.7118        | 2.1   | 300  | 0.6664          | 0.7735   |
| 0.4261        | 2.31  | 330  | 0.6058          | 0.8014   |
| 0.4394        | 2.52  | 360  | 0.5754          | 0.8223   |
| 0.4581        | 2.72  | 390  | 0.4719          | 0.8467   |
| 0.3967        | 2.93  | 420  | 0.5023          | 0.8223   |

## Contact

Any doubt, contact me on [Twitter](https://twitter.com/ehcalabres) (GitHub repo soon).


### Framework versions

- Transformers 4.8.2
- Pytorch 1.9.0+cu102
- Datasets 1.9.0
- Tokenizers 0.10.3
