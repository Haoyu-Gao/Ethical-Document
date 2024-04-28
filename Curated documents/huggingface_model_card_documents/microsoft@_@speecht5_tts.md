---
license: mit
tags:
- audio
- text-to-speech
datasets:
- libritts
---

# SpeechT5 (TTS task)

SpeechT5 model fine-tuned for speech synthesis (text-to-speech) on LibriTTS.

This model was introduced in [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

SpeechT5 was first released in [this repository](https://github.com/microsoft/SpeechT5/), [original weights](https://huggingface.co/mechanicalsea/speecht5-tts). The license used is [MIT](https://github.com/microsoft/SpeechT5/blob/main/LICENSE).



## Model Description

Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.

Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

- **Developed by:** Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.
- **Shared by [optional]:** [Matthijs Hollemans](https://huggingface.co/Matthijs)
- **Model type:** text-to-speech
- **Language(s) (NLP):** [More Information Needed]
- **License:** [MIT](https://github.com/microsoft/SpeechT5/blob/main/LICENSE)
- **Finetuned from model [optional]:** [More Information Needed]


## Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [https://github.com/microsoft/SpeechT5/]
- **Paper:** [https://arxiv.org/pdf/2110.07205.pdf]
- **Blog Post:** [https://huggingface.co/blog/speecht5]
- **Demo:** [https://huggingface.co/spaces/Matthijs/speecht5-tts-demo]


# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

## 🤗 Transformers Usage

You can run SpeechT5 TTS locally with the 🤗 Transformers library.

1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers), sentencepiece, soundfile and datasets(optional):

```
pip install --upgrade pip
pip install --upgrade transformers sentencepiece datasets[audio]
```

2. Run inference via the `Text-to-Speech` (TTS) pipeline. You can access the SpeechT5 model via the TTS pipeline in just a few lines of code!

```python
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

3. Run inference via the Transformers modelling code - You can use the processor + generate code to convert text into a mono 16 kHz speech waveform for more fine-grained control.

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
```

### Fine-tuning the Model

Refer to [this Colab notebook](https://colab.research.google.com/drive/1i7I5pzBcU3WDFarDnzweIj4-sVVoIUFJ) for an example of how to fine-tune SpeechT5 for TTS on a different dataset or a new language.


## Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

You can use this model for speech synthesis. See the [model hub](https://huggingface.co/models?search=speecht5) to look for fine-tuned versions on a task that interests you.

## Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

## Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

LibriTTS

## Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

### Preprocessing [optional]

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text.


### Training hyperparameters
- **Precision:** [More Information Needed] <!--fp16, bf16, fp8, fp32 -->
- **Regime:** [More Information Needed] <!--mixed precision or not -->

### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

## Testing Data, Factors & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

[More Information Needed]

### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

## Results

[More Information Needed]

### Summary



# Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

# Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

# Technical Specifications [optional]

## Model Architecture and Objective

The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets.

After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

## Compute Infrastructure

[More Information Needed]

### Hardware

[More Information Needed]

### Software

[More Information Needed]

# Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@inproceedings{ao-etal-2022-speecht5,
    title = {{S}peech{T}5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing},
    author = {Ao, Junyi and Wang, Rui and Zhou, Long and Wang, Chengyi and Ren, Shuo and Wu, Yu and Liu, Shujie and Ko, Tom and Li, Qing and Zhang, Yu and Wei, Zhihua and Qian, Yao and Li, Jinyu and Wei, Furu},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    month = {May},
    year = {2022},
    pages={5723--5738},
}
```

# Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

- **text-to-speech** to synthesize audio

# More Information [optional]

[More Information Needed]

# Model Card Authors [optional]

Disclaimer: The team releasing SpeechT5 did not write a model card for this model so this model card has been written by the Hugging Face team.

# Model Card Contact

[More Information Needed]



