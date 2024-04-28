---
language: en
license: apache-2.0
tags:
- speech
- audio
- hubert
- audio-classification
datasets:
- superb
widget:
- example_title: Speech Commands "down"
  src: https://cdn-media.huggingface.co/speech_samples/keyword_spotting_down.wav
- example_title: Speech Commands "go"
  src: https://cdn-media.huggingface.co/speech_samples/keyword_spotting_go.wav
---

# Hubert-Base for Keyword Spotting

## Model description

This is a ported version of [S3PRL's Hubert for the SUPERB Keyword Spotting task](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/speech_commands).

The base model is [hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960), which is pretrained on 16kHz 
sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz. 

For more information refer to [SUPERB: Speech processing Universal PERformance Benchmark](https://arxiv.org/abs/2105.01051)

## Task and dataset description

Keyword Spotting (KS) detects preregistered keywords by classifying utterances into a predefined set of 
words. The task is usually performed on-device for the fast response time. Thus, accuracy, model size, and
inference time are all crucial. SUPERB uses the widely used 
[Speech Commands dataset v1.0](https://www.tensorflow.org/datasets/catalog/speech_commands) for the task.
The dataset consists of ten classes of keywords, a class for silence, and an unknown class to include the
false positive. 

For the original model's training and evaluation instructions refer to the 
[S3PRL downstream task README](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream#ks-keyword-spotting).


## Usage examples

You can use the model via the Audio Classification pipeline:
```python
from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("anton-l/superb_demo", "ks", split="test")

classifier = pipeline("audio-classification", model="superb/hubert-base-superb-ks")
labels = classifier(dataset[0]["file"], top_k=5)
```

Or use the model directly:
```python
import torch
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from torchaudio.sox_effects import apply_effects_file

effects = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]
def map_to_array(example):
    speech, _ = apply_effects_file(example["file"], effects)
    example["speech"] = speech.squeeze(0).numpy()
    return example

# load a demo dataset and read audio files
dataset = load_dataset("anton-l/superb_demo", "ks", split="test")
dataset = dataset.map(map_to_array)

model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
```

## Eval results

The evaluation metric is accuracy.

|        | **s3prl** | **transformers** |
|--------|-----------|------------------|
|**test**| `0.9630`  | `0.9672`         |

### BibTeX entry and citation info

```bibtex
@article{yang2021superb,
  title={SUPERB: Speech processing Universal PERformance Benchmark},
  author={Yang, Shu-wen and Chi, Po-Han and Chuang, Yung-Sung and Lai, Cheng-I Jeff and Lakhotia, Kushal and Lin, Yist Y and Liu, Andy T and Shi, Jiatong and Chang, Xuankai and Lin, Guan-Ting and others},
  journal={arXiv preprint arXiv:2105.01051},
  year={2021}
}
```