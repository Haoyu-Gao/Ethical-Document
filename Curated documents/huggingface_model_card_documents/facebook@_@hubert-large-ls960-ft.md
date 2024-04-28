---
language: en
license: apache-2.0
tags:
- speech
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
datasets:
- libri-light
- librispeech_asr
model-index:
- name: hubert-large-ls960-ft
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (clean)
      type: librispeech_asr
      config: clean
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 1.9
      name: Test WER
---

# Hubert-Large-Finetuned

[Facebook's Hubert](https://ai.facebook.com/blog/hubert-self-supervised-representation-learning-for-speech-recognition-generation-and-compression)

The large model fine-tuned on 960h of Librispeech on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz. 

The model is a fine-tuned version of [hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k).

[Paper](https://arxiv.org/abs/2106.07447)

Authors: Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed

**Abstract**
Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training phase, and (3) sound units have variable lengths with no explicit segmentation. To deal with these three problems, we propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an offline clustering step to provide aligned target labels for a BERT-like prediction loss. A key ingredient of our approach is applying the prediction loss over the masked regions only, which forces the model to learn a combined acoustic and language model over the continuous inputs. HuBERT relies primarily on the consistency of the unsupervised clustering step rather than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either matches or improves upon the state-of-the-art wav2vec 2.0 performance on the Librispeech (960h) and Libri-light (60,000h) benchmarks with 10min, 1h, 10h, 100h, and 960h fine-tuning subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER reduction on the more challenging dev-other and test-other evaluation subsets.

The original model can be found under https://github.com/pytorch/fairseq/tree/master/examples/hubert .

# Usage

The model can be used for automatic-speech-recognition as follows: 

```python
import torch
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

# ->"A MAN SAID TO THE UNIVERSE SIR I EXIST"
```