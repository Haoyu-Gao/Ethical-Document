---
language:
- en
license: other
datasets:
- wikipedia
pipeline_tag: text-generation
---
## Model description
This is a LLaMA-like model with only 160M parameters trained on Wikipedia and part of the C4-en and C4-realnewslike datasets. 

No evaluation has been conducted yet, so use it with care.

The model is mainly developed as a base Small Speculative Model in the [SpecInfer](https://arxiv.org/abs/2305.09781) paper.

## Citation
To cite the model, please use
```bibtex
@misc{miao2023specinfer,
      title={SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification}, 
      author={Xupeng Miao and Gabriele Oliaro and Zhihao Zhang and Xinhao Cheng and Zeyu Wang and Rae Ying Yee Wong and Zhuoming Chen and Daiyaan Arfeen and Reyna Abhyankar and Zhihao Jia},
      year={2023},
      eprint={2305.09781},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
