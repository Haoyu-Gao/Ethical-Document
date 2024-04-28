---
license: cc-by-nc-4.0
library_name: transformers
tags:
- SeamlessM4T
- seamless_m4t
inference: true
pipeline_tag: text-to-speech
---

# SeamlessM4T Medium

SeamlessM4T is a collection of models designed to provide high quality translation, allowing people from different 
linguistic communities to communicate effortlessly through speech and text. 

This repository hosts 🤗 Hugging Face's [implementation](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t) of SeamlessM4T. You can find the original weights, as well as a guide on how to run them in the original hub repositories ([large](https://huggingface.co/facebook/seamless-m4t-large) and [medium](https://huggingface.co/facebook/seamless-m4t-medium) checkpoints).

-------------------

**🌟 SeamlessM4T v2, an improved version of this version with a novel architecture, has been released [here](https://huggingface.co/facebook/seamless-m4t-v2-large). 
This new model improves over SeamlessM4T v1 in quality as well as inference speed in speech generation tasks.**

**SeamlessM4T v2 is also supported by 🤗 Transformers, more on it [in the model card of this new version](https://huggingface.co/facebook/seamless-m4t-v2-large#transformers-usage) or directly in [🤗 Transformers docs](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t_v2).**

-------------------

SeamlessM4T Medium covers:
- 📥 101 languages for speech input
- ⌨️ [196 Languages](https://huggingface.co/ylacombe/hf-seamless-m4t-medium/blob/main/tokenizer_config.json#L1887-L2089) for text input/output
- 🗣️ [35 languages](https://huggingface.co/ylacombe/hf-seamless-m4t-medium/blob/main/generation_config.json#L253-L288) for speech output. 

This is the "medium" variant of the unified model, which enables multiple tasks without relying on multiple separate models:
- Speech-to-speech translation (S2ST)
- Speech-to-text translation (S2TT)
- Text-to-speech translation (T2ST)
- Text-to-text translation (T2TT)
- Automatic speech recognition (ASR)

You can perform all the above tasks from one single model, [`SeamlessM4TModel`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel), but each task also has its own dedicated sub-model.


## 🤗 Usage

First, load the processor and a checkpoint of the model:

```python
>>> from transformers import AutoProcessor, SeamlessM4TModel

>>> processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
>>> model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
```

You can seamlessly use this model on text or on audio, to generated either translated text or translated audio.

Here is how to use the processor to process text and audio:

```python
>>> # let's load an audio sample from an Arabic speech corpus
>>> from datasets import load_dataset
>>> dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
>>> audio_sample = next(iter(dataset))["audio"]

>>> # now, process it
>>> audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

>>> # now, process some English test as well
>>> text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```


### Speech

[`SeamlessM4TModel`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) can *seamlessly* generate text or speech with few or no changes. Let's target Russian voice translation:

```python
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

With basically the same code, I've translated English text and Arabic speech to Russian speech samples.

### Text

Similarly, you can generate translated text from audio files or from text with the same model. You only have to pass `generate_speech=False` to [`SeamlessM4TModel.generate`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel.generate).
This time, let's translate to French.

```python 
>>> # from audio
>>> output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_audio = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

>>> # from text
>>> output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
```

### Tips


#### 1. Use dedicated models

[`SeamlessM4TModel`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint.
For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code: 

```python
>>> from transformers import SeamlessM4TForSpeechToSpeech
>>> model = SeamlessM4TForSpeechToSpeech.from_pretrained("facebook/hf-seamless-m4t-medium")
```

Or you can replace the text-to-text generation snippet with the model dedicated to the T2TT task, you only have to remove `generate_speech=False`.

```python
>>> from transformers import SeamlessM4TForTextToText
>>> model = SeamlessM4TForTextToText.from_pretrained("facebook/hf-seamless-m4t-medium")
```

Feel free to try out [`SeamlessM4TForSpeechToText`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToText) and [`SeamlessM4TForTextToSpeech`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToSpeech) as well.

#### 2. Change the speaker identity

You have the possibility to change the speaker used for speech synthesis with the `spkr_id` argument. Some `spkr_id` works better than other for some languages!

#### 3. Change the generation strategy

You can use different [generation strategies](https://huggingface.co/docs/transformers/v4.34.1/en/generation_strategies#text-generation-strategies) for speech and text generation, e.g `.generate(input_ids=input_ids, text_num_beams=4, speech_do_sample=True)` which will successively perform beam-search decoding on the text model, and multinomial sampling on the speech model.

#### 4. Generate speech and text at the same time

Use `return_intermediate_token_ids=True` with [`SeamlessM4TModel`](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) to return both speech and text !