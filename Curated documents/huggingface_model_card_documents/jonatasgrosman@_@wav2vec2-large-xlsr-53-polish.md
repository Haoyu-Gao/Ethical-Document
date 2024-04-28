---
language: pl
license: apache-2.0
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
- mozilla-foundation/common_voice_6_0
- pl
- robust-speech-event
- speech
- xlsr-fine-tuning-week
datasets:
- common_voice
- mozilla-foundation/common_voice_6_0
metrics:
- wer
- cer
model-index:
- name: XLSR Wav2Vec2 Polish by Jonatas Grosman
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice pl
      type: common_voice
      args: pl
    metrics:
    - type: wer
      value: 14.21
      name: Test WER
    - type: cer
      value: 3.49
      name: Test CER
    - type: wer
      value: 10.98
      name: Test WER (+LM)
    - type: cer
      value: 2.93
      name: Test CER (+LM)
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: pl
    metrics:
    - type: wer
      value: 33.18
      name: Dev WER
    - type: cer
      value: 15.92
      name: Dev CER
    - type: wer
      value: 29.31
      name: Dev WER (+LM)
    - type: cer
      value: 15.17
      name: Dev CER (+LM)
---

# Fine-tuned XLSR-53 large model for speech recognition in Polish

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Polish using the train and validation splits of [Common Voice 6.1](https://huggingface.co/datasets/common_voice).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [HuggingSound](https://github.com/jonatasgrosman/huggingsound) library:

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-polish")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

Writing your own inference script:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "pl"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
SAMPLES = 5

test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence)
```

| Reference  | Prediction |
| ------------- | ------------- |
| """CZY DRZWI BYŁY ZAMKNIĘTE?""" | PRZY DRZWI BYŁY ZAMKNIĘTE |
| GDZIEŻ TU POWÓD DO WYRZUTÓW? | WGDZIEŻ TO POM DO WYRYDÓ |
| """O TEM JEDNAK NIE BYŁO MOWY.""" | O TEM JEDNAK NIE BYŁO MOWY |
| LUBIĘ GO. | LUBIĄ GO |
| — TO MI NIE POMAGA. | TO MNIE NIE POMAGA |
| WCIĄŻ LUDZIE WYSIADAJĄ PRZED ZAMKIEM, Z MIASTA, Z PRAGI. | WCIĄŻ LUDZIE WYSIADAJĄ PRZED ZAMKIEM Z MIASTA Z PRAGI |
| ALE ON WCALE INACZEJ NIE MYŚLAŁ. | ONY MONITCENIE PONACZUŁA NA MASU |
| A WY, CO TAK STOICIE? | A WY CO TAK STOICIE |
| A TEN PRZYRZĄD DO CZEGO SŁUŻY? | A TEN PRZYRZĄD DO CZEGO SŁUŻY |
| NA JUTRZEJSZYM KOLOKWIUM BĘDZIE PIĘĆ PYTAŃ OTWARTYCH I TEST WIELOKROTNEGO WYBORU. | NAJUTRZEJSZYM KOLOKWIUM BĘDZIE PIĘĆ PYTAŃ OTWARTYCH I TEST WIELOKROTNEGO WYBORU |

## Evaluation

1. To evaluate on `mozilla-foundation/common_voice_6_0` with split `test`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-polish --dataset mozilla-foundation/common_voice_6_0 --config pl --split test
```

2. To evaluate on `speech-recognition-community-v2/dev_data`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-polish --dataset speech-recognition-community-v2/dev_data --config pl --split validation --chunk_length_s 5.0 --stride_length_s 1.0
```

## Citation
If you want to cite this model you can use this:

```bibtex
@misc{grosman2021xlsr53-large-polish,
  title={Fine-tuned {XLSR}-53 large model for speech recognition in {P}olish},
  author={Grosman, Jonatas},
  howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish}},
  year={2021}
}
```