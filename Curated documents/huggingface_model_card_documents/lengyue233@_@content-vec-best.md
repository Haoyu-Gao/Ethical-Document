---
license: mit
---

# Content Vec Best
Official Repo: [ContentVec](https://github.com/auspicious3000/contentvec)  
This repo brings fairseq ContentVec model to HuggingFace Transformers.

## How to use
To use this model, you need to define
```python
class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
```

and then load the model with
```python
model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")

x = model(audio)["last_hidden_state"]
```

## How to convert
You need to download the ContentVec_legacy model from the official repo, and then run
```bash
python convert.py
```
