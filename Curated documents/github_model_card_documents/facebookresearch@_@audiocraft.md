# AudioGen Model Card

## Model details
**Organization developing the model:** The FAIR team of Meta AI.

**Model date:** This version of AudioGen was trained between July 2023 and August 2023.

**Model version:** This is version 2 of the model, not to be confused with the original AudioGen model published in ["AudioGen: Textually Guided Audio Generation"][audiogen].
In this version (v2), AudioGen was trained on the same data, but with some other differences:
1. This model was trained on 10 seconds (vs. 5 seconds in v1).
2. The discrete representation used under the hood is extracted using a retrained EnCodec model on the environmental sound data, following the EnCodec setup detailed in the ["Simple and Controllable Music Generation" paper][musicgen].
3. No audio mixing augmentations.

**Model type:** AudioGen consists of an EnCodec model for audio tokenization, and an auto-regressive language model based on the transformer architecture for audio modeling. The released model has 1.5B parameters.

**Paper or resource for more information:** More information can be found in the paper [AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352).

**Citation details:** See [AudioGen paper][audiogen]

**License:** Code is released under MIT, model weights are released under CC-BY-NC 4.0.

**Where to send questions or comments about the model:** Questions and comments about AudioGen can be sent via the [GitHub repository](https://github.com/facebookresearch/audiocraft) of the project, or by opening an issue.

## Intended use
**Primary intended use:** The primary use of AudioGen is research on AI-based audio generation, including:
- Research efforts, such as probing and better understanding the limitations of generative models to further improve the state of science
- Generation of sound guided by text to understand current abilities of generative AI models by machine learning amateurs

**Primary intended users:** The primary intended users of the model are researchers in audio, machine learning and artificial intelligence, as well as amateur seeking to better understand those models.

**Out-of-scope use cases** The model should not be used on downstream applications without further risk evaluation and mitigation. The model should not be used to intentionally create or disseminate audio pieces that create hostile or alienating environments for people. This includes generating audio that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Metrics

**Models performance measures:** We used the following objective measure to evaluate the model on a standard audio benchmark:
- Frechet Audio Distance computed on features extracted from a pre-trained audio classifier (VGGish)
- Kullback-Leibler Divergence on label distributions extracted from a pre-trained audio classifier (PaSST)

Additionally, we run qualitative studies with human participants, evaluating the performance of the model with the following axes:
- Overall quality of the audio samples;
- Text relevance to the provided text input;

More details on performance measures and human studies can be found in the paper.

**Decision thresholds:** Not applicable.

## Evaluation datasets

The model was evaluated on the [AudioCaps benchmark](https://audiocaps.github.io/).

## Training datasets

The model was trained on the following data sources: a subset of AudioSet (Gemmeke et al., 2017), [BBC sound effects](https://sound-effects.bbcrewind.co.uk/), AudioCaps (Kim et al., 2019), Clotho v2 (Drossos et al., 2020), VGG-Sound (Chen et al., 2020), FSD50K (Fonseca et al., 2021), [Free To Use Sounds](https://www.freetousesounds.com/all-in-one-bundle/), [Sonniss Game Effects](https://sonniss.com/gameaudiogdc), [WeSoundEffects](https://wesoundeffects.com/we-sound-effects-bundle-2020/), [Paramount Motion - Odeon Cinematic Sound Effects](https://www.paramountmotion.com/odeon-sound-effects).

## Evaluation results

Below are the objective metrics obtained with the released model on AudioCaps (consisting of 10-second long samples). Note that the model differs from the original AudioGen model introduced in the paper, hence the difference in the metrics.

| Model | Frechet Audio Distance | KLD | Text consistency |
|---|---|---|---|
| facebook/audiogen-medium | 1.77 | 1.58 | 0.30 |

More information can be found in the paper [AudioGen: Textually Guided Audio Generation][audiogen], in the Experiments section.

## Limitations and biases

**Limitations:**
- The model is not able to generate realistic vocals.
- The model has been trained with English descriptions and will not perform as well in other languages.
- It is sometimes difficult to assess what types of text descriptions provide the best generations. Prompt engineering may be required to obtain satisfying results.

**Biases:** The datasets used for training may be lacking of diversity and are not representative of all possible sound events. The generated samples from the model will reflect the biases from the training data.

**Risks and harms:** Biases and limitations of the model may lead to generation of samples that may be considered as biased, inappropriate or offensive. We believe that providing the code to reproduce the research and train new models will allow to broaden the application to new and more representative data.

**Use cases:** Users must be aware of the biases, limitations and risks of the model. AudioGen is a model developed for artificial intelligence research on audio generation. As such, it should not be used for downstream applications without further investigation and mitigation of risks.

[musicgen]: https://arxiv.org/abs/2306.05284
[audiogen]: https://arxiv.org/abs/2209.15352


# MusicGen Model Card

## Model details

**Organization developing the model:** The FAIR team of Meta AI.

**Model date:** MusicGen was trained between April 2023 and May 2023.

**Model version:** This is the version 1 of the model.

**Model type:** MusicGen consists of an EnCodec model for audio tokenization, an auto-regressive language model based on the transformer architecture for music modeling. The model comes in different sizes: 300M, 1.5B and 3.3B parameters ; and two variants: a model trained for text-to-music generation task and a model trained for melody-guided music generation.

**Paper or resources for more information:** More information can be found in the paper [Simple and Controllable Music Generation][arxiv].

**Citation details:** See [our paper][arxiv]

**License:** Code is released under MIT, model weights are released under CC-BY-NC 4.0.

**Where to send questions or comments about the model:** Questions and comments about MusicGen can be sent via the [GitHub repository](https://github.com/facebookresearch/audiocraft) of the project, or by opening an issue.

## Intended use
**Primary intended use:** The primary use of MusicGen is research on AI-based music generation, including:

- Research efforts, such as probing and better understanding the limitations of generative models to further improve the state of science
- Generation of music guided by text or melody to understand current abilities of generative AI models by machine learning amateurs

**Primary intended users:** The primary intended users of the model are researchers in audio, machine learning and artificial intelligence, as well as amateur seeking to better understand those models.

**Out-of-scope use cases:** The model should not be used on downstream applications without further risk evaluation and mitigation. The model should not be used to intentionally create or disseminate music pieces that create hostile or alienating environments for people. This includes generating music that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Metrics

**Models performance measures:** We used the following objective measure to evaluate the model on a standard music benchmark:

- Frechet Audio Distance computed on features extracted from a pre-trained audio classifier (VGGish)
- Kullback-Leibler Divergence on label distributions extracted from a pre-trained audio classifier (PaSST)
- CLAP Score between audio embedding and text embedding extracted from a pre-trained CLAP model

Additionally, we run qualitative studies with human participants, evaluating the performance of the model with the following axes:

- Overall quality of the music samples;
- Text relevance to the provided text input;
- Adherence to the melody for melody-guided music generation.

More details on performance measures and human studies can be found in the paper.

**Decision thresholds:** Not applicable.

## Evaluation datasets

The model was evaluated on the [MusicCaps benchmark](https://www.kaggle.com/datasets/googleai/musiccaps) and on an in-domain held-out evaluation set, with no artist overlap with the training set.

## Training datasets

The model was trained on licensed data using the following sources: the [Meta Music Initiative Sound Collection](https://www.fb.com/sound),  [Shutterstock music collection](https://www.shutterstock.com/music) and the [Pond5 music collection](https://www.pond5.com/). See the paper for more details about the training set and corresponding preprocessing.

## Evaluation results

Below are the objective metrics obtained on MusicCaps with the released model. Note that for the publicly released models, we had all the datasets go through a state-of-the-art music source separation method, namely using the open source [Hybrid Transformer for Music Source Separation](https://github.com/facebookresearch/demucs) (HT-Demucs), in order to keep only the instrumental part. This explains the difference in objective metrics with the models used in the paper.

| Model | Frechet Audio Distance | KLD | Text Consistency | Chroma Cosine Similarity |
|---|---|---|---|---|
| facebook/musicgen-small  | 4.88 | 1.42 | 0.27 | - |
| facebook/musicgen-medium | 5.14 | 1.38 | 0.28 | - |
| facebook/musicgen-large  | 5.48 | 1.37 | 0.28 | - |
| facebook/musicgen-melody | 4.93 | 1.41 | 0.27 | 0.44 |

More information can be found in the paper [Simple and Controllable Music Generation][arxiv], in the Results section.

## Limitations and biases

**Data:** The data sources used to train the model are created by music professionals and covered by legal agreements with the right holders. The model is trained on 20K hours of data, we believe that scaling the model on larger datasets can further improve the performance of the model.

**Mitigations:** Vocals have been removed from the data source using corresponding tags, and then using a state-of-the-art music source separation method, namely using the open source [Hybrid Transformer for Music Source Separation](https://github.com/facebookresearch/demucs) (HT-Demucs).

**Limitations:**

- The model is not able to generate realistic vocals.
- The model has been trained with English descriptions and will not perform as well in other languages.
- The model does not perform equally well for all music styles and cultures.
- The model sometimes generates end of songs, collapsing to silence.
- It is sometimes difficult to assess what types of text descriptions provide the best generations. Prompt engineering may be required to obtain satisfying results.

**Biases:** The source of data is potentially lacking diversity and all music cultures are not equally represented in the dataset. The model may not perform equally well on the wide variety of music genres that exists. The generated samples from the model will reflect the biases from the training data. Further work on this model should include methods for balanced and just representations of cultures, for example, by scaling the training data to be both diverse and inclusive.

**Risks and harms:** Biases and limitations of the model may lead to generation of samples that may be considered as biased, inappropriate or offensive. We believe that providing the code to reproduce the research and train new models will allow to broaden the application to new and more representative data.

**Use cases:** Users must be aware of the biases, limitations and risks of the model. MusicGen is a model developed for artificial intelligence research on controllable music generation. As such, it should not be used for downstream applications without further investigation and mitigation of risks.

## Update: stereo models and large melody.

We further release a set of stereophonic capable models. Those were fine tuned for 200k updates starting
from the mono models. The training data is otherwise identical and capabilities and limitations are shared with the base modes. The stereo models work by getting 2 streams of tokens from the EnCodec model, and interleaving those using
the delay pattern. We also release a mono large model with melody conditioning capabilities. The list of new models
is as follow:

- facebook/musicgen-stereo-small
- facebook/musicgen-stereo-medium
- facebook/musicgen-stereo-large
- facebook/musicgen-stereo-melody
- facebook/musicgen-melody-large
- facebook/musicgen-stereo-melody-large


[arxiv]: https://arxiv.org/abs/2306.05284