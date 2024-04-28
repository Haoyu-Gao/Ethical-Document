---
license: mit
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat-dog-music.png
  candidate_labels: playing music, playing sports
  example_title: Cat & Dog
---
# Model Card for CLIP ViT-g/14 - LAION-2B

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Training Details](#training-details)
4. [Evaluation](#evaluation)
5. [Acknowledgements](#acknowledgements) 
6. [Citation](#citation)
7. [How To Get Started With the Model](#how-to-get-started-with-the-model)


# Model Details

## Model Description

A CLIP ViT-g/14 model trained with the LAION-2B English subset of LAION-5B (https://laion.ai/blog/laion-5b/) using OpenCLIP (https://github.com/mlfoundations/open_clip).

Model training done by Romain Beaumont on the [stability.ai](https://stability.ai/) cluster. 

# Uses

As per the original [OpenAI CLIP model card](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/model-card.md), this model is intended as a research output for research communities. We hope that this model will enable researchers to better understand and explore zero-shot, arbitrary image classification. We also hope it can be used for interdisciplinary studies of the potential impact of such model. 

The OpenAI CLIP paper includes a discussion of potential downstream impacts to provide an example for this sort of analysis. Additionally, the LAION-5B blog (https://laion.ai/blog/laion-5b/) and upcoming paper include additional discussion as it relates specifically to the training dataset. 

## Direct Use

Zero-shot image classification, image and text retrieval, among others.

## Downstream Use

Image classification and other image task fine-tuning, linear probe image classification, image generation guiding and conditioning, among others.

## Out-of-Scope Use

As per the OpenAI models,

**Any** deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed use cases such as image search in a constrained environment, are also not recommended unless there is thorough in-domain testing of the model with a specific, fixed class taxonomy. This is because our safety assessment demonstrated a high need for task specific testing especially given the variability of CLIP’s performance with different class taxonomies. This makes untested and unconstrained deployment of the model in any use case currently potentially harmful. 

Certain use cases which would fall under the domain of surveillance and facial recognition are always out-of-scope regardless of performance of the model. This is because the use of artificial intelligence for tasks such as these can be premature currently given the lack of testing norms and checks to ensure its fair use.

Since the model has not been purposefully trained in or evaluated on any languages other than English, its use should be limited to English language use cases.

Further the above notice, the LAION-5B dataset used in training of these models has additional considerations, see below.

# Training Details

## Training Data

This model was trained with the 2 Billion sample English subset of LAION-5B (https://laion.ai/blog/laion-5b/).

**IMPORTANT NOTE:** The motivation behind dataset creation is to democratize research and experimentation around large-scale multi-modal model training and handling of uncurated, large-scale datasets crawled from publically available internet. Our recommendation is therefore to use the dataset for research purposes. Be aware that this large-scale dataset is uncurated. Keep in mind that the uncurated nature of the dataset means that collected links may lead to strongly discomforting and disturbing content for a human viewer. Therefore, please use the demo links with caution and at your own risk. It is possible to extract a “safe” subset by filtering out samples based on the safety tags (using a customized trained NSFW classifier that we built). While this strongly reduces the chance for encountering potentially harmful content when viewing, we cannot entirely exclude the possibility for harmful content being still present in safe mode, so that the warning holds also there. We think that providing the dataset openly to broad research and other interested communities will allow for transparent investigation of benefits that come along with training large-scale models as well as pitfalls and dangers that may stay unreported or unnoticed when working with closed large datasets that remain restricted to a small community. Providing our dataset openly, we however do not recommend using it for creating ready-to-go industrial products, as the basic research about general properties and safety of such large-scale models, which we would like to encourage with this release, is still in progress.

## Training Procedure

Please see [training notes](https://docs.google.com/document/d/1EFbMLRWSSV0LUf9Du1pWzWqgeiIRPwEWX2s1C6mAk5c) and [wandb logs](https://wandb.ai/rom1504/eval_openclip/reports/slow-g-14--VmlldzoyNTMwMjg5).

# Evaluation

Evaluation done with code in the [LAION CLIP Benchmark suite](https://github.com/LAION-AI/CLIP_benchmark).

## Testing Data, Factors & Metrics

### Testing Data

The testing is performed with VTAB+ (A combination of VTAB (https://arxiv.org/abs/1910.04867) w/ additional robustness datasets) for classification and COCO and Flickr for retrieval.

**TODO** - more detail

## Results

The model achieves a 76.6 zero-shot top-1 accuracy on ImageNet-1k.

An initial round of benchmarks have been performed on a wider range of datasets, currently viewable at https://github.com/LAION-AI/CLIP_benchmark/blob/main/benchmark/results.ipynb

**TODO** - create table for just this model's metrics.

# Acknowledgements

Acknowledging [stability.ai](https://stability.ai/) for the compute used to train this model.

# Citation

**BibTeX:**

In addition to forthcoming LAION-5B (https://laion.ai/blog/laion-5b/) paper, please cite:

OpenAI CLIP paper
```
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

OpenCLIP software
```
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

# How to Get Started with the Model

Use the code below to get started with the model.

** TODO ** - Hugging Face transformers, OpenCLIP, and timm getting started snippets