---
language:
- cs
license: apache-2.0
tags:
- automatic-speech-recognition
- generated_from_trainer
- hf-asr-leaderboard
- mozilla-foundation/common_voice_8_0
- robust-speech-event
- xlsr-fine-tuning-week
datasets:
- mozilla-foundation/common_voice_8_0
- ovm
- pscr
- vystadial2016
base_model: facebook/wav2vec2-xls-r-300m
model-index:
- name: Czech comodoro Wav2Vec2 XLSR 300M 250h data
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice 8
      type: mozilla-foundation/common_voice_8_0
      args: cs
    metrics:
    - type: wer
      value: 7.3
      name: Test WER
    - type: cer
      value: 2.1
      name: Test CER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: cs
    metrics:
    - type: wer
      value: 43.44
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Robust Speech Event - Test Data
      type: speech-recognition-community-v2/eval_data
      args: cs
    metrics:
    - type: wer
      value: 38.5
      name: Test WER
---

# Czech wav2vec2-xls-r-300m-cs-250

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the common_voice 8.0 dataset as well as other datasets listed below.

It achieves the following results on the evaluation set:
- Loss: 0.1271
- Wer: 0.1475
- Cer: 0.0329

The `eval.py` script results using a LM are:
- WER: 0.07274312090176113
- CER: 0.021207369275558875

## Model description

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Czech using the [Common Voice](https://huggingface.co/datasets/common_voice) dataset.
When using this model, make sure that your speech input is sampled at 16kHz.


The model can be used directly (without a language model) as follows:

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("mozilla-foundation/common_voice_8_0", "cs", split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained("comodoro/wav2vec2-xls-r-300m-cs-250")
model = Wav2Vec2ForCTC.from_pretrained("comodoro/wav2vec2-xls-r-300m-cs-250")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	batch["speech"] = resampler(speech_array).squeeze().numpy()
	return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset[:2]["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
	logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset[:2]["sentence"])
```

## Evaluation

The model can be evaluated using the attached `eval.py` script:
```
python eval.py --model_id comodoro/wav2vec2-xls-r-300m-cs-250 --dataset mozilla-foundation/common-voice_8_0 --split test --config cs
```

## Training and evaluation data

The Common Voice 8.0 `train` and `validation` datasets were used for training, as well as the following datasets:

- Šmídl, Luboš and Pražák, Aleš, 2013, OVM – Otázky Václava Moravce, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11858/00-097C-0000-000D-EC98-3.

- Pražák, Aleš and Šmídl, Luboš, 2012, Czech Parliament Meetings, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11858/00-097C-0000-0005-CF9C-4.

- Plátek, Ondřej; Dušek, Ondřej and Jurčíček, Filip, 2016, Vystadial 2016 – Czech data, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-1740.


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 800
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    | Cer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:------:|
| 3.4203        | 0.16  | 800   | 3.3148          | 1.0    | 1.0    |
| 2.8151        | 0.32  | 1600  | 0.8508          | 0.8938 | 0.2345 |
| 0.9411        | 0.48  | 2400  | 0.3335          | 0.3723 | 0.0847 |
| 0.7408        | 0.64  | 3200  | 0.2573          | 0.2840 | 0.0642 |
| 0.6516        | 0.8   | 4000  | 0.2365          | 0.2581 | 0.0595 |
| 0.6242        | 0.96  | 4800  | 0.2039          | 0.2433 | 0.0541 |
| 0.5754        | 1.12  | 5600  | 0.1832          | 0.2156 | 0.0482 |
| 0.5626        | 1.28  | 6400  | 0.1827          | 0.2091 | 0.0463 |
| 0.5342        | 1.44  | 7200  | 0.1744          | 0.2033 | 0.0468 |
| 0.4965        | 1.6   | 8000  | 0.1705          | 0.1963 | 0.0444 |
| 0.5047        | 1.76  | 8800  | 0.1604          | 0.1889 | 0.0422 |
| 0.4814        | 1.92  | 9600  | 0.1604          | 0.1827 | 0.0411 |
| 0.4471        | 2.09  | 10400 | 0.1566          | 0.1822 | 0.0406 |
| 0.4509        | 2.25  | 11200 | 0.1619          | 0.1853 | 0.0432 |
| 0.4415        | 2.41  | 12000 | 0.1513          | 0.1764 | 0.0397 |
| 0.4313        | 2.57  | 12800 | 0.1515          | 0.1739 | 0.0392 |
| 0.4163        | 2.73  | 13600 | 0.1445          | 0.1695 | 0.0377 |
| 0.4142        | 2.89  | 14400 | 0.1478          | 0.1699 | 0.0385 |
| 0.4184        | 3.05  | 15200 | 0.1430          | 0.1669 | 0.0376 |
| 0.3886        | 3.21  | 16000 | 0.1433          | 0.1644 | 0.0374 |
| 0.3795        | 3.37  | 16800 | 0.1426          | 0.1648 | 0.0373 |
| 0.3859        | 3.53  | 17600 | 0.1357          | 0.1604 | 0.0361 |
| 0.3762        | 3.69  | 18400 | 0.1344          | 0.1558 | 0.0349 |
| 0.384         | 3.85  | 19200 | 0.1379          | 0.1576 | 0.0359 |
| 0.3762        | 4.01  | 20000 | 0.1344          | 0.1539 | 0.0346 |
| 0.3559        | 4.17  | 20800 | 0.1339          | 0.1525 | 0.0351 |
| 0.3683        | 4.33  | 21600 | 0.1315          | 0.1518 | 0.0342 |
| 0.3572        | 4.49  | 22400 | 0.1307          | 0.1507 | 0.0342 |
| 0.3494        | 4.65  | 23200 | 0.1294          | 0.1491 | 0.0335 |
| 0.3476        | 4.81  | 24000 | 0.1287          | 0.1491 | 0.0336 |
| 0.3475        | 4.97  | 24800 | 0.1271          | 0.1475 | 0.0329 |

### Framework versions

- Transformers 4.16.2
- Pytorch 1.10.1+cu102
- Datasets 1.18.3
- Tokenizers 0.11.0
