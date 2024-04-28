---
language: fr
license: apache-2.0
tags:
- audio
- automatic-speech-recognition
- fr
- hf-asr-leaderboard
- mozilla-foundation/common_voice_6_0
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
- name: XLSR Wav2Vec2 French by Jonatas Grosman
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice fr
      type: common_voice
      args: fr
    metrics:
    - type: wer
      value: 17.65
      name: Test WER
    - type: cer
      value: 4.89
      name: Test CER
    - type: wer
      value: 13.59
      name: Test WER (+LM)
    - type: cer
      value: 3.91
      name: Test CER (+LM)
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: fr
    metrics:
    - type: wer
      value: 34.35
      name: Dev WER
    - type: cer
      value: 14.09
      name: Dev CER
    - type: wer
      value: 24.72
      name: Dev WER (+LM)
    - type: cer
      value: 12.33
      name: Dev CER (+LM)
---

# Fine-tuned XLSR-53 large model for speech recognition in French

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on French using the train and validation splits of [Common Voice 6.1](https://huggingface.co/datasets/common_voice).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [HuggingSound](https://github.com/jonatasgrosman/huggingsound) library:

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-french")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

Writing your own inference script:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
SAMPLES = 10

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
| "CE DERNIER A ÉVOLUÉ TOUT AU LONG DE L'HISTOIRE ROMAINE." | CE DERNIER ÉVOLUÉ TOUT AU LONG DE L'HISTOIRE ROMAINE |
| CE SITE CONTIENT QUATRE TOMBEAUX DE LA DYNASTIE ACHÉMÉNIDE ET SEPT DES SASSANIDES. | CE SITE CONTIENT QUATRE TOMBEAUX DE LA DYNASTIE ASHEMÉNID ET SEPT DES SASANDNIDES |
| "J'AI DIT QUE LES ACTEURS DE BOIS AVAIENT, SELON MOI, BEAUCOUP D'AVANTAGES SUR LES AUTRES." | JAI DIT QUE LES ACTEURS DE BOIS AVAIENT SELON MOI BEAUCOUP DAVANTAGES SUR LES AUTRES |
| LES PAYS-BAS ONT REMPORTÉ TOUTES LES ÉDITIONS. | LE PAYS-BAS ON REMPORTÉ TOUTES LES ÉDITIONS |
| IL Y A MAINTENANT UNE GARE ROUTIÈRE. | IL AMNARDIGAD LE TIRAN |
| HUIT | HUIT |
| DANS L’ATTENTE DU LENDEMAIN, ILS NE POUVAIENT SE DÉFENDRE D’UNE VIVE ÉMOTION | DANS L'ATTENTE DU LENDEMAIN IL NE POUVAIT SE DÉFENDRE DUNE VIVE ÉMOTION |
| LA PREMIÈRE SAISON EST COMPOSÉE DE DOUZE ÉPISODES. | LA PREMIÈRE SAISON EST COMPOSÉE DE DOUZE ÉPISODES |
| ELLE SE TROUVE ÉGALEMENT DANS LES ÎLES BRITANNIQUES. | ELLE SE TROUVE ÉGALEMENT DANS LES ÎLES BRITANNIQUES |
| ZÉRO | ZEGO |

## Evaluation

1. To evaluate on `mozilla-foundation/common_voice_6_0` with split `test`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-french --dataset mozilla-foundation/common_voice_6_0 --config fr --split test
```

2. To evaluate on `speech-recognition-community-v2/dev_data`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-french --dataset speech-recognition-community-v2/dev_data --config fr --split validation --chunk_length_s 5.0 --stride_length_s 1.0
```

## Citation
If you want to cite this model you can use this:

```bibtex
@misc{grosman2021xlsr53-large-french,
  title={Fine-tuned {XLSR}-53 large model for speech recognition in {F}rench},
  author={Grosman, Jonatas},
  howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french}},
  year={2021}
}
```