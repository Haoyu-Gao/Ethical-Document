---
language:
- en
license: cc-by-2.0
datasets:
- CreativeLang/vua20_metaphor
---

# Metaphor_Detection_Roberta_Seq

## Description

- **Paper:** [FrameBERT: Conceptual Metaphor Detection with Frame Embedding Learning](https://aclanthology.org/2023.eacl-main.114.pdf)

## Model Summary

Creative Language Toolkit (CLTK) Metadata
- CL Type: Metaphor
- Task Type: detection
- Size: roberta-base (500MB)
- Created time: 2022

This model is a easy to use metaphor detection baseline realised with `roberta-base` fine-tuned on [CreativeLang/vua20_metaphor](https://huggingface.co/datasets/CreativeLang/vua20_metaphor) dataset.

To use this model, please use the `inference.py` in the [FrameBERT repo](https://github.com/liyucheng09/MetaphorFrame).

Just run:
```
python inference.py CreativeLang/metaphor_detection_roberta_seq
```

Check out `inference.py` to learn how to apply the model on your own data.

For the details of this model and the dataset used, we refer you to the release [paper](https://aclanthology.org/2023.eacl-main.114.pdf).

## Metrics

| Metric                           | Value                    |
|----------------------------------|--------------------------|
| eval_loss                        | 0.2656                   |
| eval_accuracy_score              | 0.9142                   |
| eval_precision                   | 0.9142                   |
| eval_recall                      | 0.9142                   |
| eval_f1                          | 0.9142                   |
| eval_f1_macro                    | 0.7315                   |
| eval_runtime                     | 8.9802                   |
| eval_samples_per_second          | 411.7960                 |
| eval_steps_per_second            | 51.5580                  |
| epoch                            | 3.0000                   |


### Citation Information

If you find this dataset helpful, please cite:

```
@article{Li2023FrameBERTCM,
  title={FrameBERT: Conceptual Metaphor Detection with Frame Embedding Learning},
  author={Yucheng Li and Shunyu Wang and Chenghua Lin and Frank Guerin and Lo{\"i}c Barrault},
  journal={ArXiv},
  year={2023},
  volume={abs/2302.04834}
}
```

### Contributions

If you have any queries, please open an issue or direct your queries to [mail](mailto:yucheng.li@surrey.ac.uk).