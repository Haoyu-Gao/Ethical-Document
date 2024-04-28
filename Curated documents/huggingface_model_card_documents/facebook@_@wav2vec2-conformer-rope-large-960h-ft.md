---
language: en
license: apache-2.0
tags:
- speech
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
datasets:
- librispeech_asr
model-index:
- name: wav2vec2-conformer-rel-pos-large-960h-ft
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
      value: 1.96
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (other)
      type: librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 3.98
      name: Test WER
---

# Wav2Vec2-Conformer-Large-960h with Rotary Position Embeddings

Wav2Vec2 Conformer with rotary position embeddings, pretrained and **fine-tuned on 960 hours of Librispeech** on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz.

**Paper**: [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171)

**Authors**: Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino

The results of Wav2Vec2-Conformer can be found in Table 3 and Table 4 of the [official paper](https://arxiv.org/abs/2010.05171).

The original model can be found under https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20.


# Usage

To transcribe audio files the model can be used as a standalone acoustic model as follows:

```python
 from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
 from datasets import load_dataset
 import torch
 
 # load model and processor
 processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
 model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
     
 # load dummy dataset and read soundfiles
 ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 
 # tokenize
 input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values
 
 # retrieve logits
 logits = model(input_values).logits
 
 # take argmax and decode
 predicted_ids = torch.argmax(logits, dim=-1)
 transcription = processor.batch_decode(predicted_ids)
 ```
 
  ## Evaluation
 
 This code snippet shows how to evaluate **facebook/wav2vec2-conformer-rope-large-960h-ft** on LibriSpeech's "clean" and "other" test data.
 
```python
from datasets import load_dataset
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor
import torch
from jiwer import wer


librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred, remove_columns=["audio"])

print("WER:", wer(result["text"], result["transcription"]))
```

*Result (WER)*:

| "clean" | "other" |
|---|---|
| 1.96 | 3.98 |