---
license: cc-by-nc-4.0
tags:
- music
inference: false
---

# Introduction to our series work

The development log of our Music Audio Pre-training (m-a-p) model family:
- 02/06/2023: [arxiv pre-print](https://arxiv.org/abs/2306.00107) and training [codes](https://github.com/yizhilll/MERT) released.
- 17/03/2023: we release two advanced music understanding models, [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) and [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M) , trained with new paradigm and dataset. They outperform the previous models and can better generalize to more tasks.
- 14/03/2023: we retrained the MERT-v0 model with open-source-only music dataset [MERT-v0-public](https://huggingface.co/m-a-p/MERT-v0-public)
- 29/12/2022: a music understanding model [MERT-v0](https://huggingface.co/m-a-p/MERT-v0) trained with **MLM** paradigm, which performs better at downstream tasks.
- 29/10/2022: a pre-trained MIR model [music2vec](https://huggingface.co/m-a-p/music2vec-v1) trained with **BYOL** paradigm.



Here is a table for quick model pick-up:

| Name                                                         | Pre-train Paradigm | Training Data (hour) | Pre-train Context   (second) | Model Size | Transformer Layer-Dimension | Feature Rate | Sample Rate | Release Date |
| ------------------------------------------------------------ | ------------------ | -------------------- | ---------------------------- | ---------- | --------------------------- | ------------ | ----------- | ------------ |
| [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M)    | MLM                | 160K                 | 5                            | 330M       | 24-1024                     | 75 Hz        | 24K Hz      | 17/03/2023   |
| [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)      | MLM                | 20K                  | 5                            | 95M        | 12-768                      | 75 Hz        | 24K Hz      | 17/03/2023   |
| [MERT-v0-public](https://huggingface.co/m-a-p/MERT-v0-public) | MLM                | 900                  | 5                            | 95M        | 12-768                      | 50 Hz        | 16K Hz      | 14/03/2023   |
| [MERT-v0](https://huggingface.co/m-a-p/MERT-v0)              | MLM                | 1000                 | 5                            | 95 M       | 12-768                      | 50 Hz        | 16K Hz      | 29/12/2022   |
| [music2vec-v1](https://huggingface.co/m-a-p/music2vec-v1)    | BYOL               | 1000                 | 30                           | 95 M       | 12-768                      | 50 Hz        | 16K Hz      | 30/10/2022   |

## Explanation

The m-a-p models share the similar model architecture and the most distinguished difference is the paradigm in used pre-training. Other than that, there are several nuance technical configuration needs to know before using:

- **Model Size**: the number of parameters that would be loaded to memory. Please select the appropriate size fitting your hardware.
- **Transformer Layer-Dimension**: The number of transformer layers and the corresponding feature dimensions can be outputted from our model. This is marked out because features extracted by **different layers could have various performance depending on tasks**.
- **Feature Rate**: Given a 1-second audio input, the number of features output by the model.
- **Sample Rate**: The frequency of audio that the model is trained with.



# Introduction to MERT-v1

Compared to MERT-v0, we introduce multiple new things in the MERT-v1 pre-training:

- Change the pseudo labels to 8 codebooks from [encodec](https://github.com/facebookresearch/encodec), which potentially has higher quality and empower our model to support music generation.
- MLM prediction with in-batch noise mixture.
- Train with higher audio frequency (24K Hz).
- Train with more audio data (up to 160 thousands of hours).
- More available model sizes 95M and 330M.



More details will be written in our coming-soon paper.



# Model Usage

```python
# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset


# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)

# load demo audio and set processor
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

resample_rate = processor.sampling_rate
# make sure the sample_rate aligned
if resample_rate != sampling_rate:
    print(f'setting rate from {sampling_rate} to {resample_rate}')
    resampler = T.Resample(sampling_rate, resample_rate)
else:
    resampler = None

# audio file is decoded on the fly
if resampler is None:
	input_audio = dataset[0]["audio"]["array"]
else:
  input_audio = resampler(torch.from_numpy(dataset[0]["audio"]["array"]))
  
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# take a look at the output shape, there are 13 layers of representation
# each layer performs differently in different downstream tasks, you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

# for utterance level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [13, 768]

# you can even use a learnable weighted average representation
aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
print(weighted_avg_hidden_states.shape) # [768]
```



# Citation

```shell
@misc{li2023mert,
      title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training}, 
      author={Yizhi Li and Ruibin Yuan and Ge Zhang and Yinghao Ma and Xingran Chen and Hanzhi Yin and Chenghua Lin and Anton Ragni and Emmanouil Benetos and Norbert Gyenge and Roger Dannenberg and Ruibo Liu and Wenhu Chen and Gus Xia and Yemin Shi and Wenhao Huang and Yike Guo and Jie Fu},
      year={2023},
      eprint={2306.00107},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```