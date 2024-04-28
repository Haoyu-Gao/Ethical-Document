---
language: ko
license: apache-2.0
tags:
- speech
- audio
- automatic-speech-recognition
datasets:
- kresnik/zeroth_korean
model-index:
- name: Wav2Vec2 XLSR Korean
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Zeroth Korean
      type: kresnik/zeroth_korean
      args: clean
    metrics:
    - type: wer
      value: 4.74
      name: Test WER
    - type: cer
      value: 1.78
      name: Test CER
---


## Evaluation on Zeroth-Korean ASR corpus

[Google colab notebook(Korean)](https://colab.research.google.com/github/indra622/tutorials/blob/master/wav2vec2_korean_tutorial.ipynb)

```
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import soundfile as sf
import torch
from jiwer import wer

processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to('cuda')

ds = load_dataset("kresnik/zeroth_korean", "clean")

test_ds = ds['test']

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

test_ds = test_ds.map(map_to_array)

def map_to_pred(batch):
    inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = test_ds.map(map_to_pred, batched=True, batch_size=16, remove_columns=["speech"])

print("WER:", wer(result["text"], result["transcription"]))

```

### Expected WER: 4.74%
### Expected CER: 1.78%