---
license: mit
library_name: open_clip
tags:
- zero-shot-image-classification
- clip
pipeline_tag: zero-shot-image-classification
---
# Model card for CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Training Details](#training-details)
4. [Evaluation](#evaluation)
5. [Acknowledgements](#acknowledgements)
6. [Citation](#citation)

# Model Details

## Model Description

A series of CLIP [ConvNeXt-Large](https://arxiv.org/abs/2201.03545) (w/ extra text depth, vision MLP head) models trained on the LAION-2B (english) subset of [LAION-5B](https://arxiv.org/abs/2210.08402) using [OpenCLIP](https://github.com/mlfoundations/open_clip).

The models utilize:
  * the [timm](https://github.com/rwightman/pytorch-image-models) ConvNeXt-Large model (`convnext_large`) as the image tower
  * a MLP (`fc - gelu - drop - fc`) head in vision tower instead of the single projection of other CLIP models
  * a text tower with same width but 4 layers more depth than ViT-L / RN50x16 models (depth 16, embed dim 768).

This 320x320 resolution model is a soup (weight average) of 3 fine-tunes of [CLIP-convnext_large_d.laion2B-s26B-b102K-augreg](https://huggingface.co/laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg) at a higher resolution. It is an average of 3 fine-tunes from the final checkpoint of the original 256x256 training run w/ an additional ~2-3B samples for each fine-tune and a lower learning rate. Each fine-tune was a different learning rate (1e-4, 6e-5, 5e-5), and diff # of samples (3.2B, 2B, 2.5B).

At 320x320, the ConvNext-Large-D is significantly more efficient than the L/14 model at 336x336 that OpenAI fine-tuned. L/14-336 model is 2.5x more GMAC, 2.8x more activations, and 1.22x more parameters.

| Model | Dataset | Resolution | AugReg | Top-1 ImageNet Zero-Shot (%) |
| ----- | ------- | ---------- | ------------ | --------- |
| [convnext_large_d.laion2b_s26b_b102k-augreg](https://huggingface.co/laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg) | LAION-2B | 256x256 |  RRC (0.33, 1.0), RE (0.35), SD (0.1), D(0.1) | 75.9 |
| [convnext_large_d_320.laion2b_s29b_b131k-ft](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft) | LAION-2B | 320x320 |  RRC (0.5, 1.0), RE (0.4), SD (0.1), D(0.0) | 76.6 |
| [convnext_large_d_320.laion2b_s29b_b131k-ft-soup](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup) | LAION-2B | 320x320 |  RRC (0.5, 1.0), RE (0.4), SD (0.1), D(0.0) | 76.9 |

RRC = Random Resize Crop (crop pcts), RE = Random Erasing (prob), SD = Stochastic Depth (prob) -- image tower only, D = Dropout (prob) -- image tower head only

LAION-A = LAION Aesthetic, an ~900M sample subset of LAION-2B with pHash dedupe and asthetic score filtering.

Model training done by Ross Wightman on the [stability.ai](https://stability.ai/) cluster.

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

This model was trained with LAION-2B -- A 2 billion sample English subset of LAION-5B (https://laion.ai/blog/laion-5b/).

**IMPORTANT NOTE:** The motivation behind dataset creation is to democratize research and experimentation around large-scale multi-modal model training and handling of uncurated, large-scale datasets crawled from publically available internet. Our recommendation is therefore to use the dataset for research purposes. Be aware that this large-scale dataset is uncurated. Keep in mind that the uncurated nature of the dataset means that collected links may lead to strongly discomforting and disturbing content for a human viewer. Therefore, please use the demo links with caution and at your own risk. It is possible to extract a “safe” subset by filtering out samples based on the safety tags (using a customized trained NSFW classifier that we built). While this strongly reduces the chance for encountering potentially harmful content when viewing, we cannot entirely exclude the possibility for harmful content being still present in safe mode, so that the warning holds also there. We think that providing the dataset openly to broad research and other interested communities will allow for transparent investigation of benefits that come along with training large-scale models as well as pitfalls and dangers that may stay unreported or unnoticed when working with closed large datasets that remain restricted to a small community. Providing our dataset openly, we however do not recommend using it for creating ready-to-go industrial products, as the basic research about general properties and safety of such large-scale models, which we would like to encourage with this release, is still in progress.

## Training Procedure

All 320x320 model fine-tunes were trained with a global batch size of 131072 for 10-16 checkpoint intervals of 203.7M samples for a total of ~2-3B samples seen over fine-tune.

For 320x320 models, a slurm script w/ srun below was used on 64 8-GPU (A100 40GB) nodes (Stability).

```
/opt/slurm/sbin/srun --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --name "convnext_large_320" \
    --pretrained ""/runs/convnext_large_256/epoch_128.pt" \
    --resume 'latest' \
    --train-data="pipe:aws s3 cp s3://mybucket/path/{laion{00000..xxxxx}.tar -" \
    --train-num-samples 203666042 \
    --dataset-type webdataset \
    --precision amp_bfloat16 \
    --beta2 0.98 \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=12 \
    --dataset-resampled \
    --aug-cfg use_timm=True scale='(0.5, 1.0)' re_prob=0.4 \
    --clip-grad-norm 5.0 \
    --lr 5e-5 \
    --workers=6 \
    --model "convnext_large_d_320" \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing
```

# Evaluation

Evaluation done with code in the [LAION CLIP Benchmark suite](https://github.com/LAION-AI/CLIP_benchmark).

## Testing Data, Factors & Metrics

### Testing Data

The testing is performed with VTAB+ (A combination of VTAB (https://arxiv.org/abs/1910.04867) w/ additional robustness datasets) for classification and COCO and Flickr for retrieval.

## Results

The models achieve between 75.9 and 76.9 top-1 zero-shot accuracy on ImageNet-1k.

Zero-shot curve of origina from-scratch 256x256 training:
![](convnext_large_zero_shot.png)

An initial round of benchmarks have been performed on a wider range of datasets, to be viewable at https://github.com/LAION-AI/CLIP_benchmark/blob/main/benchmark/results.ipynb

# Acknowledgements

Acknowledging [stability.ai](https://stability.ai/) for compute used to train this model.

# Citation

**BibTeX:**

LAION-5B
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

```
@InProceedings{pmlr-v162-wortsman22a,
  title = 	 {Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author =       {Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Ya and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Kornblith, Simon and Schmidt, Ludwig},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {23965--23998},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/wortsman22a.html}
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