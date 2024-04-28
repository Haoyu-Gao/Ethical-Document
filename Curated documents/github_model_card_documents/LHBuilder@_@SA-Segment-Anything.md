# Model Card for DINOv2-S/B/L/g

These are Vision Transformer models trained following the method described in the paper:
"DINOv2: Learning Robust Visual Features without Supervision"

We provide 4 models: 1 ViT-g trained from scratch, and 3 ViT-S/B/L models distilled from the ViT-g.

## Model Details
The model takes an image as input and returns a class token and patch tokens.

The embedding dimension is: 
- 384 for ViT-S.
- 768 for ViT-B.
- 1024 for ViT-L.
- 1536 for ViT-g.

The models follow a Transformer architecture, with a patch size of 14.

For a 224x224 image, this results in 1 class token + 256 patch tokens.

The models can accept larger images provided the image shapes are multiples of the patch size (14). 
If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.

### Model Description

- **Developed by:** Meta AI
- **Model type:** Vision Transformer
- **License:** CC-BY-NC

- **Repository:** https://github.com/facebookresearch/dinov2
- **Paper:** https://arxiv.org/abs/2304.07193
- **Demo:** https://dinov2.metademolab.com/

## Uses

The models are vision backbones providing multi-purpose features for downstream tasks.

### Direct Use

The models can be used without fine-tuning, with downstream classifiers as simple as linear layers, to obtain competitive results:
- on depth estimation, semantic segmentation, using linear layers.
- on image classification, using k-NN classifiers on the class token.
- on image classification, with logistic regression classifiers applied on the class token.
- on image classification, with a linear layer applied on the class token and the average of the patch tokens.
- on image retrieval using nearest neighbors.

### Downstream Use

It is technically possible to perform fine-tuning on the models, for small gains (we measured +2% on ImageNet-1k classification). 
We recommend keeping this as a very last step and only when necessary, as the features already provide good performance out-of-the-box.

## Bias, Risks, and Limitations

Despite improvements thanks to the training method not using annotations, we still observe significant biases in our models toward rich households from Western countries.

### Recommendations

We expect fine-tuning will increase the biases in the features produced by the model as they will be tuned to the fine-tuning labels.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
```

## Training Details

### Training Data

- **Training data:** LVD-142M (see paper)
- **Training regime:** fp16 using PyTorch-FSDP mixed-precision.

### Training Procedure 

- **Training objective:**
  - DINO self-distillation loss with multi-crop
  - iBOT masked-image modeling loss
  - KoLeo regularization on [CLS] tokens
- **Architectures:**
  - ViT-S (21M params): Patch size 14, embedding dimension 384, 6 heads, MLP FFN
  - ViT-B (86M params): Patch size 14, embedding dimension 768, 12 heads, MLP FFN
  - ViT-L (0.3B params): Patch size 14, embedding dimension 1024, 16 heads, MLP FFN
  - ViT-g (1.1B params): Patch size 14, embedding dimension 1536, 24 heads, SwiGLU FFN
- **Distillation:**
  - Distillation follows the standard DINOv2 pretraining procedure, except the teacher is a pretrained ViT-g, frozen.

## Evaluation

We refer users to the associated paper for the evaluation protocols.

<table>
  <tr>
    <th>model</th>
    <th colspan="3">ImageNet-1k</th>
    <th>NYU-Depth v2</th>
    <th>SUN-RGBD</th>
    <th>ADE20k</th>
    <th>iNaturalist 2018</th>
    <th>Oxford-H</th>
  </tr>
  <tr>
    <th rowspan="2">task</th>
    <th>classif. (acc)</th>
    <th>classif. (acc)</th>
    <th>classif. V2 (acc)</th>
    <th>depth (RMSE)</th>
    <th>depth (RMSE)</th>
    <th>segm. (mAP)</th>
    <th>classif. (acc)</th>
    <th>retrieval (mAP)</th>
  </tr>
  <tr>
    <!-- <th>^</th> -->
    <th>k-NN</th>
    <th>linear</th>
    <th>linear</th>
    <th>linear<br />4 layers</th>
    <th>NYU-D transfer</th>
    <th>multiscale</th>
    <th>linear</th>
    <th>nearest neighbor</th>
  </tr>
  <tr>
    <td>ViT-S/14</td>
    <td align="right">79.0%</td>
    <td align="right">81.1%</td>
    <td align="right">70.8%</td> 
    <td align="right">0.417</td> 
    <td align="right">0.431</td> 
    <td align="right">47.2</td> 
    <td align="right">69.5%</td> 
    <td align="right">43.2</td> 
  </tr>
  <tr>
    <td>ViT-B/14</td>
    <td align="right">82.1%</td>
    <td align="right">84.5%</td>
    <td align="right">74.9%</td>
    <td align="right">0.362</td> 
    <td align="right">0.400</td> 
    <td align="right">51.3</td> 
    <td align="right">76.3%</td> 
    <td align="right">49.5</td> 
  </tr>
  <tr>
    <td>ViT-L/14</td>
    <td align="right">83.5%</td>
    <td align="right">86.3%</td>
    <td align="right">77.6%</td>
    <td align="right">0.333</td> 
    <td align="right">0.396</td> 
    <td align="right">53.1</td> 
    <td align="right">79.8%</td> 
    <td align="right">54.0</td> 
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="right">83.5%</td>
    <td align="right">86.5%</td>
    <td align="right">78.4%</td>
    <td align="right">0.298</td> 
    <td align="right">0.362</td> 
    <td align="right">53.0</td> 
    <td align="right">81.6%</td> 
    <td align="right">52.3</td> 
  </tr>
</table>

## Environmental Impact

- **Hardware Type:** Nvidia A100
- **Hours used:** 22,000 for ViT-g, 4,500 for ViT-S distillation, 5,300 for ViT-B distillation, 8,000 for ViT-L distillation
- **Cloud Provider:** Private infra
- **Compute Region:** USA
- **Carbon Emitted:** 7t CO2eq

#### Hardware

Nvidia A100 GPUs

#### Software

PyTorch 2.0,
xFormers 0.0.18

**BibTeX**

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```



# Code Llama

## **Model Details**

**Model Developers** Meta AI 

**Variations** Code Llama comes in three model sizes, and three variants: 
1) Code Llama: our base models designed for general code synthesis and understanding
2) Code Llama - Python: designed specifically for Python 
3) Code Llama - Instruct: for instruction following and safer deployment 
 
All variants are available in sizes of 7B, 13B and 34B parameters.

**Input** Models input text only.

**Output** Models output text only.

**Model Architecture** Code Llama and its variants are autoregressive language models using optimized transformer architectures. Code Llama 7B and 13B additionally support infilling text generation. All models were fine-tuned with up to 16K tokens, and support up to 100K tokens at inference time.

**Model Dates** Code Llama and its variants have been trained between January 2023 and July 2023.

**Status** This is a static model trained on an offline dataset. Future versions of Code Llama - Instruct will be released  as we improve model safety with community feedback.

**Licence** A custom commercial license is available at: [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). 

**Research Paper** More information can be found in the paper "[Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)".

**Where to send comments** Instructions on how to provide feedback or comments on the model can be found in the model [README](README.md), or by opening an issue in the GitHub repository ([https://github.com/facebookresearch/codellama/](https://github.com/facebookresearch/codellama/)).

## **Intended Use**
**Intended Use Cases** Code Llama and its variants is intended for commercial and research use in English and relevant programming languages. The base model Code Llama can be adapted for a variety of code synthesis and understanding tasks, Code Llama - Python is designed specifically to handle the Python programming language, and Code Llama - Instruct is intended to be safer to use for code assistant and generation applications.

**Out-of-Scope Uses** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English. Use in any other way that is prohibited by the Acceptable Use Policy and Licensing Agreement for Code Llama and its variants.

## **Hardware and Software**
**Training Factors**
We used custom training libraries. The training and fine-tuning of the released models have been performed Meta’s Research Super Cluster.

**Carbon Footprint** In aggregate, training all 9 Code Llama models required 400K GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W). Estimated total emissions were 65.3 tCO2eq, 100% of which were offset by Meta’s sustainability program.

**Training data**
All experiments reported here and the released models have been trained and fine-tuned using the same data as Llama 2 with different weights (see Section 2 and Table 1 in the [research paper](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) for details).
Code Llama - Instruct uses additional instruction fine-tuning data.

**Evaluation Results**
See evaluations for the main models and detailed ablations in Section 3 and safety evaluations in Section 4 of the research paper.

## **Ethical Considerations and Limitations**
Code Llama and its variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Code Llama’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Code Llama, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available available at [https://ai.meta.com/llama/responsible-user-guide](https://ai.meta.com/llama/responsible-user-guide).


