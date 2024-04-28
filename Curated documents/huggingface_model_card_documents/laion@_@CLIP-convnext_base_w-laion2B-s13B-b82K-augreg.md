---
license: mit
library_name: open_clip
tags:
- clip
pipeline_tag: zero-shot-image-classification
---
# Model Card for CLIP-convnext_base_w.laion2B-s13B-b82k-augreg

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Training Details](#training-details)
4. [Evaluation](#evaluation)
5. [Acknowledgements](#acknowledgements)
6. [Citation](#citation)

# Model Details

## Model Description

A series of CLIP [ConvNeXt-Base](https://arxiv.org/abs/2201.03545) (w/ wide embed dim) models trained on subsets LAION-5B (https://laion.ai/blog/laion-5b/) using OpenCLIP (https://github.com/mlfoundations/open_clip).

Goals:
  * Explore an alternative to ViT and ResNet (w/ AttentionPooling) CLIP models that scales well with model size and image resolution

Firsts:
  * First known ConvNeXt CLIP models trained at scale in the range of CLIP ViT-B/16 and RN50x4 models
  * First released model weights exploring increase of augmentation + regularization for image tower via adding (greater scale range of RRC, random erasing, stochastic depth)
    
The models utilize the [timm](https://github.com/rwightman/pytorch-image-models) ConvNeXt-Base model (`convnext_base`) as the image tower, and the same text tower as the RN50x4 (depth 12, embed dim 640) model from OpenAI CLIP. The base models are trained at 256x256 image resolution and roughly match the RN50x4 models on FLOPs and activation counts. The models with `320` in the name are trained at 320x320.

All models in this series were trained for 13B samples and have ImageNet Zero-Shot top-1 of >= 70.8%. Comparing to ViT-B/16 at 34B SS with zero-shot of 70.2% (68.1% for 13B SS) this suggests the ConvNeXt architecture may be more sample efficient in this range of model scale. More experiments needed to confirm.

| Model | Dataset | Resolution | AugReg | Top-1 ImageNet Zero-Shot (%) |
| ----- | ------- | ---------- | ------------ | --------- |
| [convnext_base_w.laion2b_s13b_b82k](https://huggingface.co/laion/CLIP-convnext_base_w-laion2B-s13B-b82K) | LAION-2B | 256x256 | RRC (0.9, 1.0) | 70.8 |
| [convnext_base_w.laion2b_s13b_b82k_augreg](https://huggingface.co/laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg) | LAION-2B | 256x256 | RRC (0.33, 1.0), RE (0.35), SD (0.1) | 71.5 |
| [convnext_base_w.laion_aesthetic_s13b_b82k](https://huggingface.co/laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K) | LAION-A | 256x256 | RRC (0.9, 1.0) | 71.0 |
| [convnext_base_w_320.laion_aesthetic_s13b_b82k](https://huggingface.co/laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K) | LAION-A | 320x320 | RRC (0.9, 1.0) | 71.7 |
| [convnext_base_w_320.laion_aesthetic_s13b_b82k_augreg](https://huggingface.co/laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg) | LAION-A | 320x320 | RRC (0.33, 1.0), RE (0.35), SD (0.1) | 71.3 |

RRC = Random Resize Crop (crop pcts), RE = Random Erasing (prob), SD = Stochastic Depth (prob) -- image tower only

LAION-A = LAION Aesthetic, an ~900M sample subset of LAION-2B with pHash dedupe and asthetic score filtering.

Model training done by Ross Wightman across both the [stability.ai](https://stability.ai/) cluster and the [JUWELS Booster](https://apps.fz-juelich.de/jsc/hps/juwels/booster-overview.html) supercomputer. See acknowledgements below.

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

This model was trained with one of (see table in intro):
* LAION-2B - A 2 billion sample English subset of LAION-5B (https://laion.ai/blog/laion-5b/).
* LAION-Aesthetic - A 900M sample subset of LAION-2B with pHash dedupe and asthetic score filtering

**IMPORTANT NOTE:** The motivation behind dataset creation is to democratize research and experimentation around large-scale multi-modal model training and handling of uncurated, large-scale datasets crawled from publically available internet. Our recommendation is therefore to use the dataset for research purposes. Be aware that this large-scale dataset is uncurated. Keep in mind that the uncurated nature of the dataset means that collected links may lead to strongly discomforting and disturbing content for a human viewer. Therefore, please use the demo links with caution and at your own risk. It is possible to extract a “safe” subset by filtering out samples based on the safety tags (using a customized trained NSFW classifier that we built). While this strongly reduces the chance for encountering potentially harmful content when viewing, we cannot entirely exclude the possibility for harmful content being still present in safe mode, so that the warning holds also there. We think that providing the dataset openly to broad research and other interested communities will allow for transparent investigation of benefits that come along with training large-scale models as well as pitfalls and dangers that may stay unreported or unnoticed when working with closed large datasets that remain restricted to a small community. Providing our dataset openly, we however do not recommend using it for creating ready-to-go industrial products, as the basic research about general properties and safety of such large-scale models, which we would like to encourage with this release, is still in progress.

## Training Procedure

All models were trained with a global batch size of 81920 for 64 checkpoint intervals of 203.7M samples for a total of ~13B samples seen over training.

For 256x256 models, a slurm script w/ srun below was used on 20 8-GPU (A100 40GB) nodes (Stability), switching to 40 4-GPU nodes for time on JUWELS.

```
/opt/slurm/sbin/srun --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --name "convnext_256" \
    --resume 'latest' \
    --train-data="pipe:aws s3 cp s3://mybucket/path/{laion{00000..xxxxx}.tar -" \
    --train-num-samples 203666042 \
    --dataset-type webdataset \
    --precision amp_bfloat16 \
    --warmup 10000 \
    --batch-size=512 \
    --epochs=64 \
    --dataset-resampled \
    --clip-grad-norm 5.0 \
    --lr 1e-3 \
    --workers=6 \
    --model "convnext_base_w" \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing
```

For 320x320 models, same as above but w/ 32 8-GPU nodes, local batch size 320, or 64 4-GPU nodes on JUWELs.

# Evaluation

Evaluation done with code in the [LAION CLIP Benchmark suite](https://github.com/LAION-AI/CLIP_benchmark).

## Testing Data, Factors & Metrics

### Testing Data

The testing is performed with VTAB+ (A combination of VTAB (https://arxiv.org/abs/1910.04867) w/ additional robustness datasets) for classification and COCO and Flickr for retrieval.

## Results

The models achieve between 70.8 and 71.7 zero-shot top-1 accuracy on ImageNet-1k.

![](convnext_base_w_zero_shot.png)

An initial round of benchmarks have been performed on a wider range of datasets, to be viewable at https://github.com/LAION-AI/CLIP_benchmark/blob/main/benchmark/results.ipynb

As part of exploring increased augmentation + regularization, early evalations suggest that `augreg` trained models evaluate well over a wider range of resolutions. This is especially true for the 320x320 LAION-A model, where the augreg run was lower than the non-augreg when evaluated at the train resolution of 320x320 (71.3 vs 71.7), but improves to 72.2 when evaluated at 384x384 (the non-augreg drops to 71.0 at 384x384).

# Acknowledgements

Acknowledging [stability.ai](https://stability.ai/) and the Gauss Centre for Supercomputing e.V. (http://gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

# Citation

**BibTeX:**

```bibtex
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```

OpenCLIP software
```bibtex
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

OpenAI CLIP paper
```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

```bibtex
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```