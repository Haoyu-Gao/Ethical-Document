---
language:
- en
license: cc
datasets:
- liyucheng/FrameNet_v17
---

# Frame Classification

This model is trained FrameNet v1.7. Check out the training dataset [here](https://huggingface.co/datasets/liyucheng/FrameNet_v17).

The data is loaded with `ds = dataset.load_dataset('liyucheng/FrameNet_v17', name = 'frame_label')`.

This flatten all frame annotation to specific sentences, making frame classification a sequence tagging task.

# Metrics

```
{'accuracy_score': 0.8382018348623853, 'precision': 0.8382018348623853, 'recall': 0.8382018348623853, 'micro_f1': 0.8382018348623853, 'macro_f1': 0.45824850358482677}
```

