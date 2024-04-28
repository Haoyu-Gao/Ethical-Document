---
{}
---
This checkpoint is a wav2vec2-large model that is useful for generating transcriptions with punctuation. It is intended for use in building transcriptions for TTS models, where punctuation is very important for prosody.

This model was created by fine-tuning the `facebook/wav2vec2-large-robust-ft-libri-960h` checkpoint on the [libritts](https://research.google/tools/datasets/libri-tts/) and [voxpopuli](https://github.com/facebookresearch/voxpopuli) datasets with a new vocabulary that includes punctuation.

The model gets a respectable WER of 4.45% on the librispeech validation set. The baseline, `facebook/wav2vec2-large-robust-ft-libri-960h`, got 4.3%.

Since the model was fine-tuned on clean audio, it is not well-suited for noisy audio like CommonVoice (though I may upload a checkpoint for that soon too). It still does pretty good, though.

The vocabulary is uploaded to the model hub as well `jbetker/tacotron_symbols`.

Check out my speech transcription script repo, [ocotillo](https://github.com/neonbjb/ocotillo) for usage examples: https://github.com/neonbjb/ocotillo