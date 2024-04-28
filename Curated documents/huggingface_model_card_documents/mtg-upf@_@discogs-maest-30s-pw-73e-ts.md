---
license: cc-by-nc-sa-4.0
library_name: transformers
metrics:
- roc_auc
pipeline_tag: audio-classification
---
# Model Card for discogs-maest-30s-pw-73e-ts

## Model Details

MAEST is a family of Transformer models based on [PASST](https://github.com/kkoutini/PaSST) and
focused on music analysis applications.
The MAEST models are also available for inference in the [Essentia](https://essentia.upf.edu/models.html#maest) library and for inference and training in the [official repository](https://github.com/palonso/MAEST).
You can try the MAEST interactive demo on [replicate](https://replicate.com/mtg/maest).

> Note: This model is available under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license for non-commercial applications and under proprietary license upon request.
> [Contact us](https://www.upf.edu/web/mtg/contact) for more information.

> Note: MAEST models rely on [custom code](https://huggingface.co/docs/transformers/custom_models#using-a-model-with-custom-code). Set `trust_remote_code=True` to use them within the [🤗Transformers](https://huggingface.co/docs/transformers/)' `audio-classification` pipeline.


### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Pablo Alonso
- **Shared by:** Pablo Alonso
- **Model type:** Transformer
- **License:** cc-by-nc-sa-4.0
- **Finetuned from model:** [PaSST](https://github.com/kkoutini/PaSST)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [MAEST](https://github.com/palonso/MAEST)
- **Paper:** [Efficient Supervised Training of Audio Transformers for Music Representation Learning](http://hdl.handle.net/10230/58023)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

MAEST is a music audio representation model pre-trained on the task of music style classification.
According to the evaluation reported in the original paper, it reports good performance in several downstream music analysis tasks.

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The MAEST models can make predictions for a taxonomy of 400 music styles derived from the public metadata of [Discogs](https://www.discogs.com/).

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The MAEST models have reported good performance in downstream applications related to music genre recognition, music emotion recognition, and instrument detection.
Specifically, the original paper reports that the best performance is obtained from representations extracted from intermediate layers of the model.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model has not been evaluated outside the context of music understanding applications, so we are unaware of its performance outside its intended domain.
Since the model is intended to be used within the `audio-classification` pipeline, it is important to mention that MAEST is **NOT** a general-purpose audio classification model (such as [AST](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)), so it shuold not be expected to perform well in tasks such as [AudioSet](https://research.google.com/audioset/dataset/index.html).

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The MAEST models were trained using Discogs20, an in-house [MTG](https://www.upf.edu/web/mtg) dataset derived from the public Discogs metadata. While we tried to maximize the diversity with respect to the 400 music styles covered in the dataset, we noted an overrepresentation of Western (particularly electronic) music.

## How to Get Started with the Model

The MAEST models can be used with the `audio_classification` pipeline of the `transformers` library. For example:

```python
import numpy as np
from transformers import pipeline

# audio @16kHz
audio = np.random.randn(30 * 16000)

pipe = pipeline("audio-classification", model="mtg-upf/discogs-maest-30s-pw-73e-ts")
pipe(audio)
```

```
[{'score': 0.6158794164657593, 'label': 'Electronic---Noise'},
 {'score': 0.08825448155403137, 'label': 'Electronic---Experimental'},
 {'score': 0.08772594481706619, 'label': 'Electronic---Abstract'},
 {'score': 0.03644488751888275, 'label': 'Rock---Noise'},
 {'score': 0.03272806480526924, 'label': 'Electronic---Musique Concrète'}]
```

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Our models were trained using Discogs20, [MTG](https://www.upf.edu/web/mtg) in-house dataset featuring 3.3M music tracks matched to [Discogs](https://www.discogs.com/)' metadata.

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

Most training details are detailed in the [paper](https://arxiv.org/abs/2309.16418) and [official implementation](https://github.com/palonso/MAEST/) of the model.

#### Preprocessing

MAEST models rely on mel-spectrograms originally extracted with the Essentia library, and used in several previous publications.
In Transformers, this mel-spectrogram signature is replicated to a certain extent using `audio_utils`, which have a very small (but not neglectable) impact on the predictions.

## Evaluation, Metrics, and results

The MAEST models were pre-trained in the task of music style classification, and their internal representations were evaluated via downstream MLP probes in several benchmark music understanding tasks.
Check the original [paper](https://arxiv.org/abs/2309.16418) for details.


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

- **Hardware Type:** 4 x Nvidia RTX 2080 Ti
- **Hours used:** apprx. 32
- **Carbon Emitted:** apprx. 3.46 kg CO2 eq.

*Carbon emissions estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).*

## Technical Specifications

### Model Architecture and Objective

[Audio Spectrogram Transformer (AST)](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)

### Compute Infrastructure

Local infrastructure

#### Hardware

4 x Nvidia RTX 2080 Ti

#### Software

Pytorch

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@inproceedings{alonso2023music,
  title={Efficient supervised training of audio transformers for music representation learning},
  author={Alonso-Jim{\'e}nez, Pablo and Serra, Xavier and Bogdanov, Dmitry},
  booktitle={Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR 2023)},
  year={2022},
  organization={International Society for Music Information Retrieval (ISMIR)}
}
```

**APA:**

```
Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2023). Efficient Supervised Training of Audio Transformers for Music Representation Learning. In Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR 2023)
```

## Model Card Authors

Pablo Alonso

## Model Card Contact

* Twitter: [@pablo__alonso](https://twitter.com/pablo__alonso)

* Github: [@palonso](https://github.com/palonso/)

* mail: pablo `dot` alonso `at` upf `dot` edu