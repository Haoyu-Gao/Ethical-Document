---
language: is
license: cc-by-4.0
tags:
- audio
- automatic-speech-recognition
- icelandic
- xlrs-53-icelandic
- iceland
- reykjavik
- samromur
datasets:
- language-and-voice-lab/samromur_asr
- language-and-voice-lab/samromur_children
- language-and-voice-lab/malromur_asr
- language-and-voice-lab/althingi_asr
model-index:
- name: wav2vec2-large-xlsr-53-icelandic-ep10-1000h
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Samrómur (Test)
      type: language-and-voice-lab/samromur_asr
      split: test
      args:
        language: is
    metrics:
    - type: wer
      value: 9.847
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Samrómur (Dev)
      type: language-and-voice-lab/samromur_asr
      split: validation
      args:
        language: is
    metrics:
    - type: wer
      value: 8.736
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Samrómur Children (Test)
      type: language-and-voice-lab/samromur_children
      split: test
      args:
        language: is
    metrics:
    - type: wer
      value: 9.391
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Samrómur Children (Dev)
      type: language-and-voice-lab/samromur_children
      split: validation
      args:
        language: is
    metrics:
    - type: wer
      value: 6.055
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Malrómur (Test)
      type: language-and-voice-lab/malromur_asr
      split: test
      args:
        language: is
    metrics:
    - type: wer
      value: 5.643
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Malrómur (Dev)
      type: language-and-voice-lab/malromur_asr
      split: validation
      args:
        language: is
    metrics:
    - type: wer
      value: 6.156
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Althingi (Test)
      type: language-and-voice-lab/althingi_asr
      split: test
      args:
        language: is
    metrics:
    - type: wer
      value: 11.437
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Althingi (Dev)
      type: language-and-voice-lab/althingi_asr
      split: validation
      args:
        language: is
    metrics:
    - type: wer
      value: 11.093
      name: WER
---
# wav2vec2-large-xlsr-53-icelandic-ep10-1000h

The "wav2vec2-large-xlsr-53-icelandic-ep10-1000h" is an acoustic model suitable for Automatic Speech Recognition in Icelandic. It is the result of fine-tuning the model "facebook/wav2vec2-large-xlsr-53" for 10 epochs with around 1000 hours of Icelandic data developed by the [Language and Voice Laboratory](https://huggingface.co/language-and-voice-lab). Most of the data is available at public repositories such as [LDC](https://www.ldc.upenn.edu/), [OpenSLR](https://openslr.org/) or [Clarin.is](https://clarin.is/)

The specific list of corpora used to fine-tune the model is:

- [Samrómur 21.05 (114h34m)](http://www.openslr.org/112/)
- [Samrómur Children (127h25m)](https://catalog.ldc.upenn.edu/LDC2022S11)
- [Malrómur (119hh03m)](https://clarin.is/en/resources/malromur/)
- [Althingi Parliamentary Speech (514h29m)](https://catalog.ldc.upenn.edu/LDC2021S01)
- L2-Speakers Data (125h55m) **Unpublished material**
	
The fine-tuning process was performed during December (2022) in the servers of the Language and Voice Laboratory (https://lvl.ru.is/) at Reykjavík University (Iceland) by Carlos Daniel Hernández Mena.

# Evaluation
```python
import torch
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

#Load the processor and model.
MODEL_NAME="carlosdanielhernandezmena/wav2vec2-large-xlsr-53-icelandic-ep10-1000h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

#Load the dataset
from datasets import load_dataset, load_metric, Audio
ds=load_dataset("language-and-voice-lab/samromur_children", split="test")

#Downsample to 16kHz
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

#Process the dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    #Batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["normalized_text"]).input_ids
    return batch
ds = ds.map(prepare_dataset, remove_columns=ds.column_names,num_proc=1)

#Define the evaluation metric
import numpy as np
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    #We do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

#Do the evaluation (with batch_size=1)
model = model.to(torch.device("cuda"))
def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
    return batch
results = ds.map(map_to_result,remove_columns=ds.column_names)

#Compute the overall WER now.
print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["sentence"])))
```
**Test Result**: 0.094

# BibTeX entry and citation info
*When publishing results based on these models please refer to:*
```bibtex
@misc{mena2022xlrs53icelandic,
      title={Acoustic Model in Icelandic: wav2vec2-large-xlsr-53-icelandic-ep10-1000h.}, 
      author={Hernandez Mena, Carlos Daniel},
      url={https://huggingface.co/carlosdanielhernandezmena/wav2vec2-large-xlsr-53-icelandic-ep10-1000h},
      year={2022}
}
```

# Acknowledgements

Special thanks to Jón Guðnason, head of the Language and Voice Lab for providing computational power to make this model possible. We also want to thank to the "Language Technology Programme for Icelandic 2019-2023" which is managed and coordinated by Almannarómur, and it is funded by the Icelandic Ministry of Education, Science and Culture.
