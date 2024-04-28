---
language: zh
license: apache-2.0
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
datasets:
- common_voice
metrics:
- wer
- cer
model-index:
- name: XLSR Wav2Vec2 Chinese (zh-CN) by Jonatas Grosman
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: Common Voice zh-CN
      type: common_voice
      args: zh-CN
    metrics:
    - type: wer
      value: 82.37
      name: Test WER
    - type: cer
      value: 19.03
      name: Test CER
---

# Fine-tuned XLSR-53 large model for speech recognition in Chinese

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Chinese using the train and validation splits of [Common Voice 6.1](https://huggingface.co/datasets/common_voice), [CSS10](https://github.com/Kyubyong/css10) and [ST-CMDS](http://www.openslr.org/38/).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [HuggingSound](https://github.com/jonatasgrosman/huggingsound) library:

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

Writing your own inference script:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "zh-CN"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
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
| 宋朝末年年间定居粉岭围。 | 宋朝末年年间定居分定为 |
| 渐渐行动不便 | 建境行动不片 |
| 二十一年去世。 | 二十一年去世 |
| 他们自称恰哈拉。 | 他们自称家哈<unk> |
| 局部干涩的例子包括有口干、眼睛干燥、及阴道干燥。 | 菊物干寺的例子包括有口肝眼睛干照以及阴到干<unk> |
| 嘉靖三十八年，登进士第三甲第二名。 | 嘉靖三十八年登进士第三甲第二名 |
| 这一名称一直沿用至今。 | 这一名称一直沿用是心 |
| 同时乔凡尼还得到包税合同和许多明矾矿的经营权。 | 同时桥凡妮还得到包税合同和许多民繁矿的经营权 |
| 为了惩罚西扎城和塞尔柱的结盟，盟军在抵达后将外城烧毁。 | 为了曾罚西扎城和塞尔素的节盟盟军在抵达后将外曾烧毁 |
| 河内盛产黄色无鱼鳞的鳍射鱼。 | 合类生场环色无鱼林的骑射鱼 |

## Evaluation

The model can be evaluated as follows on the Chinese (zh-CN) test data of Common Voice.

```python
import torch
import re
import librosa
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "zh-CN"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
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

In the table below I report the Word Error Rate (WER) and the Character Error Rate (CER) of the model. I ran the evaluation script described above on other models as well (on 2021-05-13). Note that the table below may show different results from those already reported, this may have been caused due to some specificity of the other evaluation scripts used.

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn | **82.37%** | **19.03%** |
| ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt | 84.01% | 20.95% |


## Citation
If you want to cite this model you can use this:

```bibtex
@misc{grosman2021xlsr53-large-chinese,
  title={Fine-tuned {XLSR}-53 large model for speech recognition in {C}hinese},
  author={Grosman, Jonatas},
  howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn}},
  year={2021}
}
```