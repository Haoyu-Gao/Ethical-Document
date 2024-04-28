---
license: cc-by-nc-4.0
tags:
- mms
- vits
pipeline_tag: text-to-speech
---

# Massively Multilingual Speech (MMS): English Text-to-Speech

This repository contains the **English (eng)** language text-to-speech (TTS) model checkpoint.

This model is part of Facebook's [Massively Multilingual Speech](https://arxiv.org/abs/2305.13516) project, aiming to
provide speech technology across a diverse range of languages. You can find more details about the supported languages
and their ISO 639-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html),
and see all MMS-TTS checkpoints on the Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts).

MMS-TTS is available in the 🤗 Transformers library from version 4.33 onwards.

## Model Details

VITS (**V**ariational **I**nference with adversarial learning for end-to-end **T**ext-to-**S**peech) is an end-to-end 
speech synthesis model that predicts a speech waveform conditional on an input text sequence. It is a conditional variational 
autoencoder (VAE) comprised of a posterior encoder, decoder, and conditional prior.

A set of spectrogram-based acoustic features are predicted by the flow-based module, which is formed of a Transformer-based
text encoder and multiple coupling layers. The spectrogram is decoded using a stack of transposed convolutional layers,
much in the same style as the HiFi-GAN vocoder. Motivated by the one-to-many nature of the TTS problem, where the same text 
input can be spoken in multiple ways, the model also includes a stochastic duration predictor, which allows the model to 
synthesise speech with different rhythms from the same input text. 

The model is trained end-to-end with a combination of losses derived from variational lower bound and adversarial training. 
To improve the expressiveness of the model, normalizing flows are applied to the conditional prior distribution. During 
inference, the text encodings are up-sampled based on the duration prediction module, and then mapped into the 
waveform using a cascade of the flow module and HiFi-GAN decoder. Due to the stochastic nature of the duration predictor,
the model is non-deterministic, and thus requires a fixed seed to generate the same speech waveform.

For the MMS project, a separate VITS checkpoint is trained on each langauge.

## Usage

MMS-TTS is available in the 🤗 Transformers library from version 4.33 onwards. To use this checkpoint, 
first install the latest version of the library:

```
pip install --upgrade transformers accelerate
```

Then, run inference with the following code-snippet:

```python
from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "some example text in the English language"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
```

The resulting waveform can be saved as a `.wav` file:

```python
import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.float().numpy())
```

Or displayed in a Jupyter Notebook / Google Colab:

```python
from IPython.display import Audio

Audio(output.numpy(), rate=model.config.sampling_rate)
```



## BibTex citation

This model was developed by Vineel Pratap et al. from Meta AI. If you use the model, consider citing the MMS paper:

```
@article{pratap2023mms,
    title={Scaling Speech Technology to 1,000+ Languages},
    author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
    journal={arXiv},
    year={2023}
}
```

## License

The model is licensed as **CC-BY-NC 4.0**.
