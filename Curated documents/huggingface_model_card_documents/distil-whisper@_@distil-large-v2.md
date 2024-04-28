---
language:
- en
license: mit
library_name: transformers
tags:
- audio
- automatic-speech-recognition
- transformers.js
widget:
- example_title: LibriSpeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: LibriSpeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
pipeline_tag: automatic-speech-recognition
---

# Distil-Whisper: distil-large-v2

Distil-Whisper was proposed in the paper [Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430).

It is a distilled version of the Whisper model that is **6 times faster**, 49% smaller, and performs 
**within 1% WER** on out-of-distribution evaluation sets. This is the repository for distil-large-v2, 
a distilled variant of [Whisper large-v2](https://huggingface.co/openai/whisper-large-v2).

| Model                                                                      | Params / M | Rel. Latency ↑ | Short-Form WER ↓ | Long-Form WER ↓ |
|----------------------------------------------------------------------------|------------|----------------|------------------|-----------------|
| [large-v2](https://huggingface.co/openai/whisper-large-v2)                 | 1550       | 1.0            | **9.1**          | 11.7            |
|                                                                            |            |                |                  |                 |
| [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)   | 756        | 5.8            | 10.1             | **11.6**        |
| [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | 394        | **6.8**        | 11.1             | 12.4            |
| [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en)   | **166**    | 5.6            | 12.1             | 12.8            |

**Note:** Distil-Whisper is currently only available for English speech recognition. We are working with the community 
to distill Whisper on other languages. If you are interested in distilling Whisper in your language, check out the 
provided [training code](https://github.com/huggingface/distil-whisper/tree/main/training). We will update the 
[Distil-Whisper repository](https://github.com/huggingface/distil-whisper/) with multilingual checkpoints when ready!

## Usage

Distil-Whisper is supported in Hugging Face 🤗 Transformers from version 4.35 onwards. To run the model, first 
install the latest version of the Transformers library. For this example, we'll also install 🤗 Datasets to load toy 
audio dataset from the Hugging Face Hub:

```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate datasets[audio]
```

### Short-Form Transcription

The model can be used with the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
class to transcribe short-form audio files (< 30-seconds) as follows:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:
```diff
- result = pipe(sample)
+ result = pipe("audio.mp3")
```

### Long-Form Transcription

Distil-Whisper uses a chunked algorithm to transcribe long-form audio files (> 30-seconds). In practice, this chunked long-form algorithm 
is 9x faster than the sequential algorithm proposed by OpenAI in the Whisper paper (see Table 7 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)).

To enable chunking, pass the `chunk_length_s` parameter to the `pipeline`. For Distil-Whisper, a chunk length of 15-seconds
is optimal. To activate batching, pass the argument `batch_size`:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

<!---
**Tip:** The pipeline can also be used to transcribe an audio file from a remote URL, for example:

```python
result = pipe("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")
```
--->

### Speculative Decoding

Distil-Whisper can be used as an assistant model to Whisper for [speculative decoding](https://huggingface.co/blog/whisper-speculative-decoding). 
Speculative decoding mathematically ensures the exact same outputs as Whisper are obtained while being 2 times faster. 
This makes it the perfect drop-in replacement for existing Whisper pipelines, since the same outputs are guaranteed.

In the following code-snippet, we load the assistant Distil-Whisper model standalone to the main Whisper pipeline. We then
specify it as the "assistant model" for generation:

```python
from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

assistant_model_id = "distil-whisper/distil-large-v2"

assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
assistant_model.to(device)

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    generate_kwargs={"assistant_model": assistant_model},
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

## Additional Speed & Memory Improvements

You can apply additional speed and memory improvements to Distil-Whisper which we cover in the following.

### Flash Attention

We recommend using [Flash-Attention 2](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2) if your GPU allows for it.
To do so, you first need to install [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```
pip install flash-attn --no-build-isolation
```

and then all you have to do is to pass `use_flash_attention_2=True` to `from_pretrained`:

```diff
- model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)
```

### Torch Scale-Product-Attention (SDPA)

If your GPU does not support Flash Attention, we recommend making use of [BetterTransformers](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#bettertransformer).
To do so, you first need to install optimum:

```
pip install --upgrade optimum
```

And then convert your model to a "BetterTransformer" model before using it:

```diff
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = model.to_bettertransformer()
```

### Running Distil-Whisper in `openai-whisper`

To use the model in the original Whisper format, first ensure you have the [`openai-whisper`](https://pypi.org/project/openai-whisper/) package installed:

```bash
pip install --upgrade openai-whisper
```

The following code-snippet demonstrates how to transcribe a sample file from the LibriSpeech dataset loaded using 
🤗 Datasets:

```python
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from whisper import load_model, transcribe

distil_large_v2 = hf_hub_download(repo_id="distil-whisper/distil-large-v2", filename="original-model.bin")
model = load_model(distil_large_v2)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]["array"]
sample = torch.from_numpy(sample).float()

pred_out = transcribe(model, audio=sample)
print(pred_out["text"])
```

To transcribe a local audio file, simply pass the path to the audio file as the `audio` argument to transcribe:

```python
pred_out = transcribe(model, audio="audio.mp3")
```

### Whisper.cpp

Distil-Whisper can be run from the [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) repository with the original 
sequential long-form transcription algorithm. In a [provisional benchmark](https://github.com/ggerganov/whisper.cpp/pull/1424#issuecomment-1793513399) 
on Mac M1, `distil-large-v2` is 2x faster than `large-v2`, while performing to within 0.1% WER over long-form audio.

Note that future releases of Distil-Whisper will target faster CPU inference more! By distilling smaller encoders, we 
aim to achieve similar speed-ups to what we obtain on GPU.

Steps for getting started:
1. Clone the Whisper.cpp repository:
```
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
```
2. Download the ggml weights for `distil-medium.en` from the Hugging Face Hub:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='distil-whisper/distil-large-v2', filename='ggml-large-32-2.en.bin', local_dir='./models')"
```

Note that if you do not have the `huggingface_hub` package installed, you can also download the weights with `wget`:

```bash
wget https://huggingface.co/distil-whisper/distil-large-v2/resolve/main/ggml-large-32-2.en.bin -P ./models
```

3. Run inference using the provided sample audio:

```bash
make -j && ./main -m models/ggml-large-32-2.en.bin -f samples/jfk.wav
```


### Transformers.js

```js
import { pipeline } from '@xenova/transformers';

let transcriber = await pipeline('automatic-speech-recognition', 'distil-whisper/distil-large-v2');

let url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
let output = await transcriber(url);
// { text: " And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country." }
```

See the [docs](https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.AutomaticSpeechRecognitionPipeline) for more information.

*Note:* Due to the large model size, we recommend running this model server-side with [Node.js](https://huggingface.co/docs/transformers.js/guides/node-audio-processing) (instead of in-browser).

### Candle

Through an integration with Hugging Face [Candle](https://github.com/huggingface/candle/tree/main) 🕯️, Distil-Whisper is 
now available in the Rust library 🦀

Benefit from:
* Optimised CPU backend with optional MKL support for x86 and Accelerate for Macs 
* CUDA backend for efficiently running on GPUs, multiple GPU distribution via NCCL
* WASM support: run Distil-Whisper in a browser

Steps for getting started:
1. Install [`candle-core`](https://github.com/huggingface/candle/tree/main/candle-core) as explained [here](https://huggingface.github.io/candle/guide/installation.html)
2. Clone the `candle` repository locally:
```
git clone https://github.com/huggingface/candle.git
```
3. Enter the example directory for [Whisper](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper):
```
cd candle/candle-examples/examples/whisper
```
4. Run an example:
```
cargo run --example whisper --release -- --model distil-large-v2
```
5. To specify your own audio file, add the `--input` flag:
```
cargo run --example whisper --release -- --model distil-large-v2 --input audio.wav
```

### 8bit & 4bit Quantization

Coming soon ...

### Whisper.cpp

Coming soon ...

## Model Details

Distil-Whisper inherits the encoder-decoder architecture from Whisper. The encoder maps a sequence of speech vector 
inputs to a sequence of hidden-state vectors. The decoder auto-regressively predicts text tokens, conditional on all 
previous tokens and the encoder hidden-states. Consequently, the encoder is only run forward once, whereas the decoder 
is run as many times as the number of tokens generated. In practice, this means the decoder accounts for over 90% of 
total inference time. Thus, to optimise for latency, the focus should be on minimising the inference time of the decoder.

To distill the Whisper model, we reduce the number of decoder layers while keeping the encoder fixed. 
The encoder (shown in green) is entirely copied from the teacher to the student and frozen during training. 
The student's decoder consists of only two decoder layers, which are initialised from the first and last decoder layer of 
the teacher (shown in red). All other decoder layers of the teacher are discarded. The model is then trained on a weighted sum 
of the KL divergence and pseudo-label loss terms.

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/architecture.png?raw=true" width="600"/>
</p>

## Evaluation

The following code-snippets demonstrates how to evaluate the Distil-Whisper model on the LibriSpeech validation.clean 
dataset with [streaming mode](https://huggingface.co/blog/audio-datasets#streaming-mode-the-silver-bullet), meaning no 
audio data has to be downloaded to your local device.

First, we need to install the required packages, including 🤗 Datasets to stream and load the audio data, and 🤗 Evaluate to 
perform the WER calculation:

```bash
pip install --upgrade pip
pip install --upgrade transformers datasets[audio] evaluate jiwer
```

Evaluation can then be run end-to-end with the following example: 

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from datasets import load_dataset
from evaluate import load
import torch
from tqdm import tqdm

# define our torch configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

# load the model + processor
model =  AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True, low_cpu_mem_usage=True)
model = model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# load the dataset with streaming mode
dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

# define the evaluation metric
wer_metric = load("wer")
normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)

def inference(batch):
    # 1. Pre-process the audio data to log-mel spectrogram inputs
    audio = [sample["array"] for sample in batch["audio"]]
    input_features = processor(audio, sampling_rate=batch["audio"][0]["sampling_rate"], return_tensors="pt").input_features
    input_features = input_features.to(device, dtype=torch_dtype)
    
    # 2. Auto-regressively generate the predicted token ids
    pred_ids = model.generate(input_features, max_new_tokens=128, language="en", task="transcribe")
    
    # 3. Decode the token ids to the final transcription
    batch["transcription"] = processor.batch_decode(pred_ids, skip_special_tokens=True)
    batch["reference"] = batch["text"]
    return batch

dataset = dataset.map(function=inference, batched=True, batch_size=16)

all_transcriptions = []
all_references = []

# iterate over the dataset and run inference
for i, result in tqdm(enumerate(dataset), desc="Evaluating..."):
    all_transcriptions.append(result["transcription"])
    all_references.append(result["reference"])

# normalize predictions and references
all_transcriptions = [normalizer(transcription) for transcription in all_transcriptions]
all_references = [normalizer(reference) for reference in all_references]

# compute the WER metric
wer = 100 * wer_metric.compute(predictions=all_transcriptions, references=all_references)
print(wer)

```
**Print Output:**
```
2.983685535968466
```

## Intended Use

Distil-Whisper is intended to be a drop-in replacement for Whisper on English speech recognition. In particular, it 
achieves comparable WER results over out-of-distribution test data, while being 6x faster over both short and long-form 
audio.

## Data

Distil-Whisper is trained on 22,000 hours of audio data from 9 open-source, permissively licensed speech datasets on the 
Hugging Face Hub:

| Dataset                                                                                 | Size / h | Speakers | Domain                      | Licence         |
|-----------------------------------------------------------------------------------------|----------|----------|-----------------------------|-----------------|
| [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)             | 12,000   | unknown  | Internet Archive            | CC-BY-SA-4.0    |
| [Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) | 3,000    | unknown  | Narrated Wikipedia          | CC0-1.0         |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)                    | 2,500    | unknown  | Audiobook, podcast, YouTube | apache-2.0      |
| Fisher                                                                                  | 1,960    | 11,900   | Telephone conversations     | LDC             |
| [LibriSpeech](https://huggingface.co/datasets/librispeech_asr)                          | 960      | 2,480    | Audiobooks                  | CC-BY-4.0       |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                         | 540      | 1,310    | European Parliament         | CC0             |
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium)                                | 450      | 2,030    | TED talks                   | CC-BY-NC-ND 3.0 |
| SwitchBoard                                                                             | 260      | 540      | Telephone conversations     | LDC             |
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami)                                | 100      | unknown  | Meetings                    | CC-BY-4.0       |
||||||
| **Total**                                                                               | 21,770   | 18,260+  |                             |                 |

The combined dataset spans 10 distinct domains and over 50k speakers. The diversity of this dataset is crucial to ensuring 
the distilled model is robust to audio distributions and noise. 

The audio data is then pseudo-labelled using the Whisper large-v2 model: we use Whisper to generate predictions for all 
the audio in our training set and use these as the target labels during training. Using pseudo-labels ensures that the 
transcriptions are consistently formatted across datasets and provides sequence-level distillation signal during training.

## WER Filter

The Whisper pseudo-label predictions are subject to mis-transcriptions and hallucinations. To ensure we only train on 
accurate pseudo-labels, we employ a simple WER heuristic during training. First, we normalise the Whisper pseudo-labels
and the ground truth labels provided by each dataset. We then compute the WER between these labels. If the WER exceeds
a specified threshold, we discard the training example. Otherwise, we keep it for training.

Section 9.2 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430) demonstrates the effectiveness of this filter for improving downstream performance
of the distilled model. We also partially attribute Distil-Whisper's robustness to hallucinations to this filter.

## Training

The model was trained for 80,000 optimisation steps (or eight epochs). The Tensorboard training logs can be found under: https://huggingface.co/distil-whisper/distil-large-v2/tensorboard?params=scalars#frame

## Results

The distilled model performs to within 1% WER of Whisper on out-of-distribution (OOD) short-form audio, and outperforms Whisper
by 0.1% on OOD long-form audio. This performance gain is attributed to lower hallucinations.

For a detailed per-dataset breakdown of the evaluation results, refer to Tables 16 and 17 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)

Distil-Whisper is also evaluated on the [ESB benchmark](https://arxiv.org/abs/2210.13352) datasets as part of the [OpenASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard),
where it performs to within 0.2% WER of Whisper.

## Reproducing Distil-Whisper

Training and evaluation code to reproduce Distil-Whisper is available under the Distil-Whisper repository: https://github.com/huggingface/distil-whisper/tree/main/training


## License

Distil-Whisper inherits the [MIT license](https://github.com/huggingface/distil-whisper/blob/main/LICENSE) from OpenAI's Whisper model.

## Citation

If you use this model, please consider citing the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430):
```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements
* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v2) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for Cloud TPU v4s
* [`@rsonavane`](https://huggingface.co/rsonavane/distil-whisper-large-v2-8-ls) for releasing an early iteration of Distil-Whisper on the LibriSpeech dataset
