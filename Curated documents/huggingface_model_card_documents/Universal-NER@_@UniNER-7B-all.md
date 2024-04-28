---
language:
- en
license: cc-by-nc-4.0
---

---


# UniNER-7B-all

**Description**: This model is the best UniNER model. It is trained on the combinations of three data splits: (1) ChatGPT-generated [Pile-NER-type data](https://huggingface.co/datasets/Universal-NER/Pile-NER-type), (2) ChatGPT-generated [Pile-NER-definition data](https://huggingface.co/datasets/Universal-NER/Pile-NER-definition), and (3) 40 supervised datasets in the Universal NER benchmark (see Fig. 4 in paper), where we randomly sample up to 10K instances from the train split of each dataset. Note that CrossNER and MIT datasets are excluded from training for OOD evaluation.

Check our [paper](https://arxiv.org/abs/2308.03279) for more information. Check our [repo](https://github.com/universal-ner/universal-ner) about how to use the model.

## Inference
The template for inference instances is as follows:
<div style="background-color: #f6f8fa; padding: 20px; border-radius: 10px; border: 1px solid #e1e4e8; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
<strong>Prompting template:</strong><br/>
A virtual assistant answers questions from a user based on the provided text.<br/>
USER: Text: <span style="color: #d73a49;">{Fill the input text here}</span><br/>
ASSISTANT: I’ve read this text.<br/>
USER: What describes <span style="color: #d73a49;">{Fill the entity type here}</span> in the text?<br/>
ASSISTANT: <span style="color: #0366d6;">(model's predictions in JSON format)</span><br/>
</div>

### Note: Inferences are based on one entity type at a time. For multiple entity types, create separate instances for each type.

## License

This model and its associated data are released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. They are primarily used for research purposes.

## Citation

```bibtex
@article{zhou2023universalner,
      title={UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition}, 
      author={Wenxuan Zhou and Sheng Zhang and Yu Gu and Muhao Chen and Hoifung Poon},
      year={2023},
      eprint={2308.03279},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```