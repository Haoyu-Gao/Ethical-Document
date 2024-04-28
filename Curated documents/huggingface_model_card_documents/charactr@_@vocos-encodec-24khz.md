---
license: mit
---

# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

[Audio samples](https://charactr-platform.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

Vocos is a fast neural vocoder designed to synthesize audio waveforms from acoustic features. Trained using a Generative
Adversarial Network (GAN) objective, Vocos can generate waveforms in a single forward pass. Unlike other typical
GAN-based vocoders, Vocos does not model audio samples in the time domain. Instead, it generates spectral
coefficients, facilitating rapid audio reconstruction through inverse Fourier transform.

## Installation

To use Vocos only in inference mode, install it using:

```bash
pip install vocos
```

If you wish to train the model, install it with additional dependencies:

```bash
pip install vocos[train]
```

## Usage

### Reconstruct audio from EnCodec tokens

Additionally, you need to provide a `bandwidth_id` which corresponds to the embedding for bandwidth from the
list: `[1.5, 3.0, 6.0, 12.0]`.

```python
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

audio_tokens = torch.randint(low=0, high=1024, size=(8, 200))  # 8 codeboooks, 200 frames
features = vocos.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([2])  # 6 kbps

audio = vocos.decode(features, bandwidth_id=bandwidth_id)
```

Copy-synthesis from a file: It extracts and quantizes features with EnCodec, then reconstructs them with Vocos in a
single forward pass.

```python
y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

y_hat = vocos(y, bandwidth_id=bandwidth_id)
```

## Citation

If this code contributes to your research, please cite our work:

```
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```

## License

The code in this repository is released under the MIT license.