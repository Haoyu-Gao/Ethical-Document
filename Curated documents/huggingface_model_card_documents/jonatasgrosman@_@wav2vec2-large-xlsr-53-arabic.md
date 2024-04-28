---
language: ar
license: apache-2.0
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
datasets:
- common_voice
- arabic_speech_corpus
metrics:
- wer
- cer
model-index:
- name: XLSR Wav2Vec2 Arabic by Jonatas Grosman
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: Common Voice ar
      type: common_voice
      args: ar
    metrics:
    - type: wer
      value: 39.59
      name: Test WER
    - type: cer
      value: 18.18
      name: Test CER
---

# Fine-tuned XLSR-53 large model for speech recognition in Arabic

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Arabic using the train and validation splits of [Common Voice 6.1](https://huggingface.co/datasets/common_voice) and [Arabic Speech Corpus](https://huggingface.co/datasets/arabic_speech_corpus).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [HuggingSound](https://github.com/jonatasgrosman/huggingsound) library:

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

Writing your own inference script:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "ar"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
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
| ألديك قلم ؟ | ألديك قلم |
| ليست هناك مسافة على هذه الأرض أبعد من يوم أمس. | ليست نالك مسافة على هذه الأرض أبعد من يوم الأمس  م |
| إنك تكبر المشكلة. | إنك تكبر المشكلة |
| يرغب أن يلتقي بك. | يرغب أن يلتقي بك |
| إنهم لا يعرفون لماذا حتى. | إنهم لا يعرفون لماذا حتى |
| سيسعدني مساعدتك أي وقت تحب. | سيسئدنيمساعدتك أي وقد تحب |
| أَحَبُّ نظريّة علمية إليّ هي أن حلقات زحل مكونة بالكامل من الأمتعة المفقودة. | أحب نظرية علمية إلي  هي أن حل قتزح المكوينا بالكامل من الأمت عن المفقودة |
| سأشتري له قلماً. | سأشتري له قلما |
| أين المشكلة ؟ | أين المشكل |
| وَلِلَّهِ يَسْجُدُ مَا فِي السَّمَاوَاتِ وَمَا فِي الْأَرْضِ مِنْ دَابَّةٍ وَالْمَلَائِكَةُ وَهُمْ لَا يَسْتَكْبِرُونَ | ولله يسجد ما في السماوات وما في الأرض من دابة والملائكة وهم لا يستكبرون |

## Evaluation

The model can be evaluated as follows on the Arabic test data of Common Voice.

```python
import torch
import re
import librosa
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "ar"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                  "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                  "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                  "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                  "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")

wer = load_metric("wer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/wer.py
cer = load_metric("cer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/cer.py

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

predictions = [x.upper() for x in result["pred_strings"]]
references = [x.upper() for x in result["sentence"]]

print(f"WER: {wer.compute(predictions=predictions, references=references, chunk_size=1000) * 100}")
print(f"CER: {cer.compute(predictions=predictions, references=references, chunk_size=1000) * 100}")
```

**Test Result**:

In the table below I report the Word Error Rate (WER) and the Character Error Rate (CER) of the model. I ran the evaluation script described above on other models as well (on 2021-05-14). Note that the table below may show different results from those already reported, this may have been caused due to some specificity of the other evaluation scripts used.

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| jonatasgrosman/wav2vec2-large-xlsr-53-arabic | **39.59%** | **18.18%** |
| bakrianoo/sinai-voice-ar-stt | 45.30% | 21.84% |
| othrif/wav2vec2-large-xlsr-arabic | 45.93% | 20.51% |
| kmfoda/wav2vec2-large-xlsr-arabic | 54.14% | 26.07% |
| mohammed/wav2vec2-large-xlsr-arabic | 56.11% | 26.79% |
| anas/wav2vec2-large-xlsr-arabic | 62.02% | 27.09% |
| elgeish/wav2vec2-large-xlsr-53-arabic | 100.00% | 100.56% |

## Citation
If you want to cite this model you can use this:

```bibtex
@misc{grosman2021xlsr53-large-arabic,
  title={Fine-tuned {XLSR}-53 large model for speech recognition in {A}rabic},
  author={Grosman, Jonatas},
  howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic}},
  year={2021}
}
```