---
license: mit
tags:
- pyannote
- pyannote-audio
- pyannote-audio-pipeline
- audio
- voice
- speech
- speaker
- voice-activity-detection
- automatic-speech-recognition
datasets:
- ami
- dihard
- voxconverse
extra_gated_prompt: The collected information will help acquire a better knowledge
  of pyannote.audio userbase and help its maintainers apply for grants to improve
  it further. If you are an academic researcher, please cite the relevant papers in
  your own publications using the model. If you work for a company, please consider
  contributing back to pyannote.audio development (e.g. through unrestricted gifts).
  We also provide scientific consulting services around speaker diarization and machine
  listening.
extra_gated_fields:
  Company/university: text
  Website: text
  I plan to use this model for (task, type of audio data, etc): text
---

I propose (paid) scientific [consulting services](https://herve.niderb.fr/consulting.html) to companies willing to make the most of their data and open-source speech processing toolkits (and `pyannote` in particular). 

# 🎹 Voice activity detection

Relies on pyannote.audio 2.1: see [installation instructions](https://github.com/pyannote/pyannote-audio#installation).


```python
# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained voice activity detection pipeline

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")
output = pipeline("audio.wav")

for speech in output.get_timeline().support():
    # active speech between speech.start and speech.end
    ...
```


## Citation

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Address = {Brno, Czech Republic},
  Month = {August},
  Year = {2021},
}
```

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```
