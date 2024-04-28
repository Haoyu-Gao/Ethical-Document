---
language:
- en
- es
license: mit
tags:
- generated_from_trainer
datasets:
- cartesinus/iva_mt_wslot
metrics:
- bleu
pipeline_tag: translation
base_model: facebook/m2m100_418M
model-index:
- name: iva_mt_wslot-m2m100_418M-en-es
  results:
  - task:
      type: text2text-generation
      name: Sequence-to-sequence Language Modeling
    dataset:
      name: iva_mt_wslot
      type: iva_mt_wslot
      config: en-es
      split: validation
      args: en-es
    metrics:
    - type: bleu
      value: 69.2836
      name: Bleu
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# iva_mt_wslot-m2m100_418M-en-es

This model is a fine-tuned version of [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) on the iva_mt_wslot dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0115
- Bleu: 69.2836
- Gen Len: 20.2064

## Model description

More information needed

## How to use

First please make sure to install `pip install transformers`. First download model: 

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

def translate(input_text, lang):
    input_ids = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(**input_ids, forced_bos_token_id=tokenizer.get_lang_id(lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

model_name = "cartesinus/iva_mt_wslot-m2m100_418M-0.1.0-en-es"
tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang="en", tgt_lang="es")
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
```

Then you can translate either plain text like this:
```python
print(translate("set the temperature on my thermostat", "es"))
```
or you can translate with slot annotations that will be restored in tgt language:
```python
print(translate("wake me up at <a>nine am<a> on <b>friday<b>", "es"))
```
Limitations of translation with slot transfer:
1) Annotated words must be placed between semi-xml tags like this "this is \<a\>example\<a\>"
2) There is no closing tag for example "\<\a\>" in the above example - this is done on purpose to omit problems with backslash escape
3) If the sentence consists of more than one slot then simply use the next alphabet letter. For example "this is \<a\>example\<a\> with more than \<b\>one\<b\> slot"
4) Please do not add space before the first or last annotated word because this particular model was trained this way and it most probably will lower its results

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 7
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Bleu    | Gen Len |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|:-------:|
| 0.0135        | 1.0   | 2104  | 0.0122          | 66.8284 | 20.2851 |
| 0.009         | 2.0   | 4208  | 0.0112          | 68.1164 | 20.1501 |
| 0.0067        | 3.0   | 6312  | 0.0110          | 68.256  | 20.0603 |
| 0.0051        | 4.0   | 8416  | 0.0110          | 68.7002 | 20.1219 |
| 0.0037        | 5.0   | 10520 | 0.0112          | 68.699  | 20.2733 |
| 0.0027        | 6.0   | 12624 | 0.0113          | 68.9916 | 20.209  |
| 0.0023        | 7.0   | 14728 | 0.0115          | 69.2836 | 20.2064 |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu118
- Datasets 2.11.0
- Tokenizers 0.13.3

## Citation

If you use this model, please cite the following:
```
@article{Sowanski2023SlotLI,
  title={Slot Lost in Translation? Not Anymore: A Machine Translation Model for Virtual Assistants with Type-Independent Slot Transfer},
  author={Marcin Sowanski and Artur Janicki},
  journal={2023 30th International Conference on Systems, Signals and Image Processing (IWSSIP)},
  year={2023},
  pages={1-5}
}
```