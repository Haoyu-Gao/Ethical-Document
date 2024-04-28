---
language: ru
license: apache-2.0
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
- mozilla-foundation/common_voice_6_0
- robust-speech-event
- ru
- speech
- xlsr-fine-tuning-week
datasets:
- common_voice
- mozilla-foundation/common_voice_6_0
metrics:
- wer
- cer
model-index:
- name: XLSR Wav2Vec2 Russian by Jonatas Grosman
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice ru
      type: common_voice
      args: ru
    metrics:
    - type: wer
      value: 13.3
      name: Test WER
    - type: cer
      value: 2.88
      name: Test CER
    - type: wer
      value: 9.57
      name: Test WER (+LM)
    - type: cer
      value: 2.24
      name: Test CER (+LM)
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: ru
    metrics:
    - type: wer
      value: 40.22
      name: Dev WER
    - type: cer
      value: 14.8
      name: Dev CER
    - type: wer
      value: 33.61
      name: Dev WER (+LM)
    - type: cer
      value: 13.5
      name: Dev CER (+LM)
---

# Fine-tuned XLSR-53 large model for speech recognition in Russian

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Russian using the train and validation splits of [Common Voice 6.1](https://huggingface.co/datasets/common_voice) and [CSS10](https://github.com/Kyubyong/css10).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [HuggingSound](https://github.com/jonatasgrosman/huggingsound) library:

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

Writing your own inference script:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "ru"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
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
| ОН РАБОТАТЬ, А ЕЕ НЕ УДЕРЖАТЬ НИКАК — БЕГАЕТ ЗА КЛЁШЕМ КАЖДОГО БУЛЬВАРНИКА. | ОН РАБОТАТЬ А ЕЕ НЕ УДЕРЖАТ НИКАК  БЕГАЕТ ЗА КЛЕШОМ КАЖДОГО БУЛЬБАРНИКА |
| ЕСЛИ НЕ БУДЕТ ВОЗРАЖЕНИЙ, Я БУДУ СЧИТАТЬ, ЧТО АССАМБЛЕЯ СОГЛАСНА С ЭТИМ ПРЕДЛОЖЕНИЕМ. | ЕСЛИ НЕ БУДЕТ ВОЗРАЖЕНИЙ Я БУДУ СЧИТАТЬ ЧТО АССАМБЛЕЯ СОГЛАСНА С ЭТИМ ПРЕДЛОЖЕНИЕМ |
| ПАЛЕСТИНЦАМ НЕОБХОДИМО СНАЧАЛА УСТАНОВИТЬ МИР С ИЗРАИЛЕМ, А ЗАТЕМ ДОБИВАТЬСЯ ПРИЗНАНИЯ ГОСУДАРСТВЕННОСТИ. | ПАЛЕСТИНЦАМ НЕОБХОДИМО СНАЧАЛА УСТАНОВИТЬ С НИ МИР ФЕЗРЕЛЕМ А ЗАТЕМ ДОБИВАТЬСЯ ПРИЗНАНИЯ ГОСУДАРСТВЕНСКИ |
| У МЕНЯ БЫЛО ТАКОЕ ЧУВСТВО, ЧТО ЧТО-ТО ТАКОЕ ОЧЕНЬ ВАЖНОЕ Я ПРИБАВЛЯЮ. | У МЕНЯ БЫЛО ТАКОЕ ЧУВСТВО ЧТО ЧТО-ТО ТАКОЕ ОЧЕНЬ ВАЖНОЕ Я ПРЕДБАВЛЯЕТ |
| ТОЛЬКО ВРЯД ЛИ ПОЙМЕТ. | ТОЛЬКО ВРЯД ЛИ ПОЙМЕТ |
| ВРОНСКИЙ, СЛУШАЯ ОДНИМ УХОМ, ПЕРЕВОДИЛ БИНОКЛЬ С БЕНУАРА НА БЕЛЬ-ЭТАЖ И ОГЛЯДЫВАЛ ЛОЖИ. | ЗЛАЗКИ СЛУШАЮ ОТ ОДНИМ УХАМ ТЫ ВОТИ В ВИНОКОТ СПИЛА НА ПЕРЕТАЧ И ОКЛЯДЫВАЛ БОСУ |
| К СОЖАЛЕНИЮ, СИТУАЦИЯ ПРОДОЛЖАЕТ УХУДШАТЬСЯ. | К СОЖАЛЕНИЮ СИТУАЦИИ ПРОДОЛЖАЕТ УХУЖАТЬСЯ |
| ВСЁ ЖАЛОВАНИЕ УХОДИЛО НА ДОМАШНИЕ РАСХОДЫ И НА УПЛАТУ МЕЛКИХ НЕПЕРЕВОДИВШИХСЯ ДОЛГОВ. | ВСЕ ЖАЛОВАНИЕ УХОДИЛО НА ДОМАШНИЕ РАСХОДЫ И НА УПЛАТУ МЕЛКИХ НЕ ПЕРЕВОДИВШИХСЯ ДОЛГОВ |
| ТЕПЕРЬ ДЕЛО, КОНЕЧНО, ЗА ТЕМ, ЧТОБЫ ПРЕВРАТИТЬ СЛОВА В ДЕЛА. | ТЕПЕРЬ ДЕЛАЮ КОНЕЧНО ЗАТЕМ ЧТОБЫ ПРЕВРАТИТЬ СЛОВА В ДЕЛА |
| ДЕВЯТЬ | ЛЕВЕТЬ |

## Evaluation

1. To evaluate on `mozilla-foundation/common_voice_6_0` with split `test`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-russian --dataset mozilla-foundation/common_voice_6_0 --config ru --split test
```

2. To evaluate on `speech-recognition-community-v2/dev_data`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-russian --dataset speech-recognition-community-v2/dev_data --config ru --split validation --chunk_length_s 5.0 --stride_length_s 1.0
```

## Citation
If you want to cite this model you can use this:

```bibtex
@misc{grosman2021xlsr53-large-russian,
  title={Fine-tuned {XLSR}-53 large model for speech recognition in {R}ussian},
  author={Grosman, Jonatas},
  howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian}},
  year={2021}
}
```