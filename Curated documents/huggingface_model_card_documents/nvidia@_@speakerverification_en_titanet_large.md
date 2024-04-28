---
language:
- en
license: cc-by-4.0
library_name: nemo
tags:
- speaker
- speech
- audio
- speaker-verification
- speaker-recognition
- speaker-diarization
- titanet
- NeMo
- pytorch
datasets:
- VOXCELEB-1
- VOXCELEB-2
- FISHER
- switchboard
- librispeech_asr
- SRE
widget:
- src: https://huggingface.co/nvidia/speakerverification_en_titanet_large/resolve/main/an255-fash-b.wav
  example_title: Speech sample 1
- src: https://huggingface.co/nvidia/speakerverification_en_titanet_large/resolve/main/cen7-fash-b.wav
  example_title: Speech sample 2
model-index:
- name: speakerverification_en_titanet_large
  results:
  - task:
      type: speaker-verification
      name: Speaker Verification
    dataset:
      name: voxceleb1
      type: voxceleb1-O
      config: clean
      split: test
      args:
        language: en
    metrics:
    - type: eer
      value: 0.66
      name: Test EER
  - task:
      type: Speaker Diarization
      name: speaker-diarization
    dataset:
      name: ami-mixheadset
      type: ami_diarization
      config: oracle-vad-known-number-of-speakers
      split: test
      args:
        language: en
    metrics:
    - type: der
      value: 1.73
      name: Test DER
    - type: der
      value: 2.03
      name: Test DER
  - task:
      type: Speaker Diarization
      name: speaker-diarization
    dataset:
      name: ch109
      type: callhome_diarization
      config: oracle-vad-known-number-of-speakers
      split: test
      args:
        language: en
    metrics:
    - type: der
      value: 1.19
      name: Test DER
  - task:
      type: Speaker Diarization
      name: speaker-diarization
    dataset:
      name: nist-sre-2000
      type: nist-sre_diarization
      config: oracle-vad-known-number-of-speakers
      split: test
      args:
        language: en
    metrics:
    - type: der
      value: 6.73
      name: Test DER
---

# NVIDIA TitaNet-Large (en-US)

<style>
img {
 display: inline;
}
</style>

| [![Model architecture](https://img.shields.io/badge/Model_Arch-TitaNet--Large-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-23M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-en--US-lightgrey#model-badge)](#datasets)


This model extracts speaker embeddings from given speech, which is the backbone for speaker verification and diarization tasks.
It is a "large" version of TitaNet (around 23M parameters) models.  
See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#titanet) for complete architecture details.

## NVIDIA NeMo: Training

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed the latest Pytorch version.
```
pip install nemo_toolkit['all']
``` 

## How to Use this Model

The model is available for use in the NeMo toolkit [3] and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
```

### Embedding Extraction

Using 

```python
emb = speaker_model.get_embedding("an255-fash-b.wav")
```

### Verifying two utterances (Speaker Verification)

Now to check if two audio files are from the same speaker or not, simply do:

```python
speaker_model.verify_speakers("an255-fash-b.wav","cen7-fash-b.wav")
```

### Extracting Embeddings for more audio files

To extract embeddings from a bunch of audio files:

Write audio files to a `manifest.json` file with lines as in format:

```json
{"audio_filepath": "<absolute path to dataset>/audio_file.wav", "duration": "duration of file in sec", "label": "speaker_id"}
```

Then running following script will extract embeddings and writes to current working directory:
```shell
python <NeMo_root>/examples/speaker_tasks/recognition/extract_speaker_embeddings.py --manifest=manifest.json
```

### Input

This model accepts 16000 KHz Mono-channel Audio (wav files) as input.

### Output

This model provides speaker embeddings for an audio file. 

## Model Architecture

TitaNet model is a depth-wise separable conv1D model [1] for Speaker Verification and diarization tasks. You may find more info on the detail of this model here: [TitaNet-Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/models.html). 

## Training

The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/recognition/speaker_reco.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/recognition/conf/titanet-large.yaml).

### Datasets

All the models in this collection are trained on a composite dataset comprising several thousand hours of English speech:

- Voxceleb-1
- Voxceleb-2
- Fisher
- Switchboard
- Librispeech
- SRE (2004-2010) 

## Performance

Performances of the these models are reported in terms of Equal Error Rate (EER%) on speaker verification evaluation trial files and as Diarization Error Rate (DER%) on diarization test sessions.

* Speaker Verification (EER%)
| Version | Model | Model Size | VoxCeleb1 (Cleaned trial file) |
|---------|--------------|-----|---------------|
| 1.10.0 | TitaNet-Large | 23M | 0.66   |

* Speaker Diarization (DER%)
| Version | Model | Model Size | Evaluation Condition | NIST SRE 2000 | AMI (Lapel) | AMI (MixHeadset) | CH109 |
|---------|--------------|-----|----------------------|---------------|-------------|------------------|-------|
| 1.10.0 | TitaNet-Large | 23M | Oracle VAD KNOWN # of Speakers  |      6.73     |      2.03      |         1.73        |  1.19 |
| 1.10.0 | TitaNet-Large | 23M | Oracle VAD UNKNOWN # of Speakers  |    5.38     |      2.03      |         1.89        |  1.63 |

## Limitations
This model is trained on both telephonic and non-telephonic speech from voxceleb datasets, Fisher and switch board. If your domain of data differs from trained data or doesnot show relatively good performance consider finetuning for that speech domain.

## NVIDIA Riva: Deployment

[NVIDIA Riva](https://developer.nvidia.com/riva), is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded. 
Additionally, Riva provides: 

* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours 
* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization 
* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support 

Although this model isn’t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  
Check out [Riva live demo](https://developer.nvidia.com/riva#demos). 

## References
[1] [TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable convolutions and global context](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9746806) 
[2] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.