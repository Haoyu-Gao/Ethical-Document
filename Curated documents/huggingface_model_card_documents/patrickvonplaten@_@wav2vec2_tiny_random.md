---
{}
---
## Test model

To test this model run the following code:

```python
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC
import torchaudio
import torch

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2_tiny_random")

def load_audio(batch):
    batch["samples"], _ = torchaudio.load(batch["file"])
    return batch
    
ds = ds.map(load_audio)

input_values = torch.nn.utils.rnn.pad_sequence([torch.tensor(x[0]) for x in ds["samples"][:10]], batch_first=True)

# forward
logits = model(input_values).logits
pred_ids = torch.argmax(logits, dim=-1)

# dummy loss
dummy_labels = pred_ids.clone()
dummy_labels[dummy_labels == model.config.pad_token_id] = 1  # can't have CTC blank token in label
dummy_labels = dummy_labels[:, -(dummy_labels.shape[1] // 4):] # make sure labels are shorter to avoid "inf" loss (can still happen though...)
loss = model(input_values, labels=dummy_labels).loss
```
