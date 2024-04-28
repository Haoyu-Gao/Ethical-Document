---
language:
- en
license: cc-by-2.0
tags:
- text2image
- prompting
datasets:
- succinctly/midjourney-prompts
thumbnail: https://drive.google.com/uc?export=view&id=1JWwrxQbr1s5vYpIhPna_p2IG1pE5rNiV
---

This is a GPT-2 model fine-tuned on the [succinctly/midjourney-prompts](https://huggingface.co/datasets/succinctly/midjourney-prompts) dataset, which contains 250k text prompts that users issued to the [Midjourney](https://www.midjourney.com/) text-to-image service over a month period. For more details on how this dataset was scraped, see [Midjourney User Prompts & Generated Images (250k)](https://www.kaggle.com/datasets/succinctlyai/midjourney-texttoimage).

This prompt generator can be used to auto-complete prompts for any text-to-image model (including the DALL·E family):
![prompt autocomplete model](https://drive.google.com/uc?export=view&id=1JqZ-CaWNpQ4iO0Qcd3b8u_QnBp-Q0PKu)


Note that, while this model can be used together with any text-to-image model, it occasionally produces Midjourney-specific tags. Users can specify certain requirements via [double-dashed parameters](https://midjourney.gitbook.io/docs/imagine-parameters) (e.g. `--ar 16:9` sets the aspect ratio to 16:9, and `--no snake` asks the model to exclude snakes from the generated image) or set the importance of various entities in the image via [explicit weights](https://midjourney.gitbook.io/docs/user-manual#advanced-text-weights) (e.g. `hot dog::1.5 food::-1` is likely to produce the image of an animal instead of a frankfurter).


When using this model, please attribute credit to [Succinctly AI](https://succinctly.ai).