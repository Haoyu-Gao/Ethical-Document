---
inference: false
---

![encodec image](https://github.com/facebookresearch/encodec/raw/2d29d9353c2ff0ab1aeadc6a3d439854ee77da3e/architecture.png)
# Model Card for EnCodec

This model card provides details and information about EnCodec, a state-of-the-art real-time audio codec developed by Meta AI.

## Model Details

### Model Description

EnCodec is a high-fidelity audio codec leveraging neural networks. It introduces a streaming encoder-decoder architecture with quantized latent space, trained in an end-to-end fashion. 
The model simplifies and speeds up training using a single multiscale spectrogram adversary that efficiently reduces artifacts and produces high-quality samples. 
It also includes a novel loss balancer mechanism that stabilizes training by decoupling the choice of hyperparameters from the typical scale of the loss. 
Additionally, lightweight Transformer models are used to further compress the obtained representation while maintaining real-time performance.

- **Developed by:** Meta AI
- **Model type:** Audio Codec

### Model Sources

- **Repository:** [GitHub Repository](https://github.com/facebookresearch/encodec)
- **Paper:** [EnCodec: End-to-End Neural Audio Codec](https://arxiv.org/abs/2210.13438)

## Uses
<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

EnCodec can be used directly as an audio codec for real-time compression and decompression of audio signals. 
It provides high-quality audio compression and efficient decoding. The model was trained on various bandwiths, which can be specified when encoding (compressing) and decoding (decompressing).
Two different setup exist for EnCodec: 

- Non-streamable: the input audio is split into chunks of 1 seconds, with an overlap of 10 ms, which are then encoded.
- Streamable: weight normalizationis used on the convolution layers, and the input is not split into chunks but rather padded on the left.

### Downstream Use

EnCodec can be fine-tuned for specific audio tasks or integrated into larger audio processing pipelines for applications such as speech generation,
music generation, or text to speech tasks.

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

## How to Get Started with the Model

Use the following code to get started with the EnCodec model using a dummy example from the LibriSpeech dataset (~9MB). First, install the required Python packages:

```
pip install --upgrade pip
pip install --upgrade datasets[audio]
pip install git+https://github.com/huggingface/transformers.git@main
```

Then load an audio sample, and run a forward pass of the model:

```python
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor


# load a demonstration datasets
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[0]["audio"]["array"]

# pre-process the inputs
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

## Training Details

The model was trained for 300 epochs, with one epoch being 2,000 updates with the Adam optimizer with a batch size of 64 examples of 1 second each, a learning rate of 3 · 10−4
, β1 = 0.5, and β2 = 0.9. All the models are traind using 8 A100 GPUs.

### Training Data


<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- For speech:
  - DNS Challenge 4
  - [Common Voice](https://huggingface.co/datasets/common_voice)
- For general audio:
  - [AudioSet](https://huggingface.co/datasets/Fhrozen/AudioSet2K22)
  - [FSD50K](https://huggingface.co/datasets/Fhrozen/FSD50k)
- For music:
  - [Jamendo dataset](https://huggingface.co/datasets/rkstgr/mtg-jamendo)


They used four different training strategies to sample for these datasets: 

- (s1) sample a single source from Jamendo with probability 0.32;
- (s2) sample a single source from the other datasets with the same probability;
- (s3) mix two sources from all datasets with a probability of 0.24;
- (s4) mix three sources from all datasets except music with a probability of 0.12.

The audio is normalized by file and a random gain between -10 and 6 dB id applied. 

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Subjectif metric for restoration:

This models was evalutated using the MUSHRA protocol (Series, 2014), using both a hidden reference and a low anchor. Annotators were recruited using a
crowd-sourcing platform, in which they were asked to rate the perceptual quality of the provided samples in
a range between 1 to 100. They randomly select 50 samples of 5 seconds from each category of the the test set
and force at least 10 annotations per samples. To filter noisy annotations and outliers we remove annotators
who rate the reference recordings less then 90 in at least 20% of the cases, or rate the low-anchor recording
above 80 more than 50% of the time. 

### Objective metric for restoration:
The ViSQOL()ink) metric was used together with the Scale-Invariant Signal-to-Noise Ration (SI-SNR) (Luo & Mesgarani, 2019;
Nachmani et al., 2020; Chazan et al., 2021).

### Results

The results of the evaluation demonstrate the superiority of EnCodec compared to the baselines across different bandwidths (1.5, 3, 6, and 12 kbps). 

When comparing EnCodec with the baselines at the same bandwidth, EnCodec consistently outperforms them in terms of MUSHRA score. 
Notably, EnCodec achieves better performance, on average, at 3 kbps compared to Lyra-v2 at 6 kbps and Opus at 12 kbps. 
Additionally, by incorporating the language model over the codes, it is possible to achieve a bandwidth reduction of approximately 25-40%. 
For example, the bandwidth of the 3 kbps model can be reduced to 1.9 kbps.


#### Summary

EnCodec is a state-of-the-art real-time neural audio compression model that excels in producing high-fidelity audio samples at various sample rates and bandwidths. 
The model's performance was evaluated across different settings, ranging from 24kHz monophonic at 1.5 kbps to 48kHz stereophonic, showcasing both subjective and 
objective results. Notably, EnCodec incorporates a novel spectrogram-only adversarial loss, effectively reducing artifacts and enhancing sample quality. 
Training stability and interpretability were further enhanced through the introduction of a gradient balancer for the loss weights. 
Additionally, the study demonstrated that a compact Transformer model can be employed to achieve an additional bandwidth reduction of up to 40% without compromising 
quality, particularly in applications where low latency is not critical (e.g., music streaming).


## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@misc{défossez2022high,
      title={High Fidelity Neural Audio Compression}, 
      author={Alexandre Défossez and Jade Copet and Gabriel Synnaeve and Yossi Adi},
      year={2022},
      eprint={2210.13438},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
