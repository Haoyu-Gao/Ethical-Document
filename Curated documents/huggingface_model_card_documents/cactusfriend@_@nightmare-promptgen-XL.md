---
license: openrail
library_name: transformers
pipeline_tag: text-generation
widget:
- text: a photograph of
  example_title: photo
- text: a bizarre cg render
  example_title: render
- text: the spaghetti
  example_title: meal?
- text: a (detailed+ intricate)+ picture
  example_title: weights
- text: photograph of various
  example_title: variety
inference:
  parameters:
    temperature: 2.6
    max_new_tokens: 250
---
Experimental 'XL' version of [Nightmare InvokeAI Prompts](https://huggingface.co/cactusfriend/nightmare-invokeai-prompts). Very early version and may be deleted.