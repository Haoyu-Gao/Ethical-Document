---
license: cc-by-nc-4.0
tags:
- image-classification
- timm
library_tag: timm
---
# Model card for resnext101_32x16d.fb_swsl_ig1b_ft_in1k

A ResNeXt-B image classification model.

This model features:
 * ReLU activations
 * single layer 7x7 convolution with pooling
 * 1x1 convolution shortcut downsample
 * grouped 3x3 bottleneck convolutions

Pretrained on Instagram-1B hashtags dataset using semi-weakly supervised learning and fine-tuned on ImageNet-1k by paper authors.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 194.0
  - GMACs: 36.3
  - Activations (M): 51.2
  - Image size: 224 x 224
- **Papers:**
  - Billion-scale semi-supervised learning for image classification: https://arxiv.org/abs/1905.00546
  - Aggregated Residual Transformations for Deep Neural Networks: https://arxiv.org/abs/1611.05431
  - Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
- **Original:** https://github.com/facebookresearch/semi-supervised-ImageNet1K-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('resnext101_32x16d.fb_swsl_ig1b_ft_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 64, 112, 112])
    #  torch.Size([1, 256, 56, 56])
    #  torch.Size([1, 512, 28, 28])
    #  torch.Size([1, 1024, 14, 14])
    #  torch.Size([1, 2048, 7, 7])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 2048, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

|model                                     |img_size|top1 |top5 |param_count|gmacs|macts|img/sec|
|------------------------------------------|--------|-----|-----|-----------|-----|-----|-------|
|[seresnextaa101d_32x8d.sw_in12k_ft_in1k_288](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288)|320     |86.72|98.17|93.6       |35.2 |69.7 |451    |
|[seresnextaa101d_32x8d.sw_in12k_ft_in1k_288](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288)|288     |86.51|98.08|93.6       |28.5 |56.4 |560    |
|[seresnextaa101d_32x8d.sw_in12k_ft_in1k](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k)|288     |86.49|98.03|93.6       |28.5 |56.4 |557    |
|[seresnextaa101d_32x8d.sw_in12k_ft_in1k](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k)|224     |85.96|97.82|93.6       |17.2 |34.2 |923    |
|[resnext101_32x32d.fb_wsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext101_32x32d.fb_wsl_ig1b_ft_in1k)|224     |85.11|97.44|468.5      |87.3 |91.1 |254    |
|[resnetrs420.tf_in1k](https://huggingface.co/timm/resnetrs420.tf_in1k)|416     |85.0 |97.12|191.9      |108.4|213.8|134    |
|[ecaresnet269d.ra2_in1k](https://huggingface.co/timm/ecaresnet269d.ra2_in1k)|352     |84.96|97.22|102.1      |50.2 |101.2|291    |
|[ecaresnet269d.ra2_in1k](https://huggingface.co/timm/ecaresnet269d.ra2_in1k)|320     |84.73|97.18|102.1      |41.5 |83.7 |353    |
|[resnetrs350.tf_in1k](https://huggingface.co/timm/resnetrs350.tf_in1k)|384     |84.71|96.99|164.0      |77.6 |154.7|183    |
|[seresnextaa101d_32x8d.ah_in1k](https://huggingface.co/timm/seresnextaa101d_32x8d.ah_in1k)|288     |84.57|97.08|93.6       |28.5 |56.4 |557    |
|[resnetrs200.tf_in1k](https://huggingface.co/timm/resnetrs200.tf_in1k)|320     |84.45|97.08|93.2       |31.5 |67.8 |446    |
|[resnetrs270.tf_in1k](https://huggingface.co/timm/resnetrs270.tf_in1k)|352     |84.43|96.97|129.9      |51.1 |105.5|280    |
|[seresnext101d_32x8d.ah_in1k](https://huggingface.co/timm/seresnext101d_32x8d.ah_in1k)|288     |84.36|96.92|93.6       |27.6 |53.0 |595    |
|[seresnet152d.ra2_in1k](https://huggingface.co/timm/seresnet152d.ra2_in1k)|320     |84.35|97.04|66.8       |24.1 |47.7 |610    |
|[resnetrs350.tf_in1k](https://huggingface.co/timm/resnetrs350.tf_in1k)|288     |84.3 |96.94|164.0      |43.7 |87.1 |333    |
|[resnext101_32x8d.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext101_32x8d.fb_swsl_ig1b_ft_in1k)|224     |84.28|97.17|88.8       |16.5 |31.2 |1100   |
|[resnetrs420.tf_in1k](https://huggingface.co/timm/resnetrs420.tf_in1k)|320     |84.24|96.86|191.9      |64.2 |126.6|228    |
|[seresnext101_32x8d.ah_in1k](https://huggingface.co/timm/seresnext101_32x8d.ah_in1k)|288     |84.19|96.87|93.6       |27.2 |51.6 |613    |
|[resnext101_32x16d.fb_wsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext101_32x16d.fb_wsl_ig1b_ft_in1k)|224     |84.18|97.19|194.0      |36.3 |51.2 |581    |
|[resnetaa101d.sw_in12k_ft_in1k](https://huggingface.co/timm/resnetaa101d.sw_in12k_ft_in1k)|288     |84.11|97.11|44.6       |15.1 |29.0 |1144   |
|[resnet200d.ra2_in1k](https://huggingface.co/timm/resnet200d.ra2_in1k)|320     |83.97|96.82|64.7       |31.2 |67.3 |518    |
|[resnetrs200.tf_in1k](https://huggingface.co/timm/resnetrs200.tf_in1k)|256     |83.87|96.75|93.2       |20.2 |43.4 |692    |
|[seresnextaa101d_32x8d.ah_in1k](https://huggingface.co/timm/seresnextaa101d_32x8d.ah_in1k)|224     |83.86|96.65|93.6       |17.2 |34.2 |923    |
|[resnetrs152.tf_in1k](https://huggingface.co/timm/resnetrs152.tf_in1k)|320     |83.72|96.61|86.6       |24.3 |48.1 |617    |
|[seresnet152d.ra2_in1k](https://huggingface.co/timm/seresnet152d.ra2_in1k)|256     |83.69|96.78|66.8       |15.4 |30.6 |943    |
|[seresnext101d_32x8d.ah_in1k](https://huggingface.co/timm/seresnext101d_32x8d.ah_in1k)|224     |83.68|96.61|93.6       |16.7 |32.0 |986    |
|[resnet152d.ra2_in1k](https://huggingface.co/timm/resnet152d.ra2_in1k)|320     |83.67|96.74|60.2       |24.1 |47.7 |706    |
|[resnetrs270.tf_in1k](https://huggingface.co/timm/resnetrs270.tf_in1k)|256     |83.59|96.61|129.9      |27.1 |55.8 |526    |
|[seresnext101_32x8d.ah_in1k](https://huggingface.co/timm/seresnext101_32x8d.ah_in1k)|224     |83.58|96.4 |93.6       |16.5 |31.2 |1013   |
|[resnetaa101d.sw_in12k_ft_in1k](https://huggingface.co/timm/resnetaa101d.sw_in12k_ft_in1k)|224     |83.54|96.83|44.6       |9.1  |17.6 |1864   |
|[resnet152.a1h_in1k](https://huggingface.co/timm/resnet152.a1h_in1k)|288     |83.46|96.54|60.2       |19.1 |37.3 |904    |
|[resnext101_32x16d.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k)|224     |83.35|96.85|194.0      |36.3 |51.2 |582    |
|[resnet200d.ra2_in1k](https://huggingface.co/timm/resnet200d.ra2_in1k)|256     |83.23|96.53|64.7       |20.0 |43.1 |809    |
|[resnext101_32x4d.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext101_32x4d.fb_swsl_ig1b_ft_in1k)|224     |83.22|96.75|44.2       |8.0  |21.2 |1814   |
|[resnext101_64x4d.c1_in1k](https://huggingface.co/timm/resnext101_64x4d.c1_in1k)|288     |83.16|96.38|83.5       |25.7 |51.6 |590    |
|[resnet152d.ra2_in1k](https://huggingface.co/timm/resnet152d.ra2_in1k)|256     |83.14|96.38|60.2       |15.4 |30.5 |1096   |
|[resnet101d.ra2_in1k](https://huggingface.co/timm/resnet101d.ra2_in1k)|320     |83.02|96.45|44.6       |16.5 |34.8 |992    |
|[ecaresnet101d.miil_in1k](https://huggingface.co/timm/ecaresnet101d.miil_in1k)|288     |82.98|96.54|44.6       |13.4 |28.2 |1077   |
|[resnext101_64x4d.tv_in1k](https://huggingface.co/timm/resnext101_64x4d.tv_in1k)|224     |82.98|96.25|83.5       |15.5 |31.2 |989    |
|[resnetrs152.tf_in1k](https://huggingface.co/timm/resnetrs152.tf_in1k)|256     |82.86|96.28|86.6       |15.6 |30.8 |951    |
|[resnext101_32x8d.tv2_in1k](https://huggingface.co/timm/resnext101_32x8d.tv2_in1k)|224     |82.83|96.22|88.8       |16.5 |31.2 |1099   |
|[resnet152.a1h_in1k](https://huggingface.co/timm/resnet152.a1h_in1k)|224     |82.8 |96.13|60.2       |11.6 |22.6 |1486   |
|[resnet101.a1h_in1k](https://huggingface.co/timm/resnet101.a1h_in1k)|288     |82.8 |96.32|44.6       |13.0 |26.8 |1291   |
|[resnet152.a1_in1k](https://huggingface.co/timm/resnet152.a1_in1k)|288     |82.74|95.71|60.2       |19.1 |37.3 |905    |
|[resnext101_32x8d.fb_wsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext101_32x8d.fb_wsl_ig1b_ft_in1k)|224     |82.69|96.63|88.8       |16.5 |31.2 |1100   |
|[resnet152.a2_in1k](https://huggingface.co/timm/resnet152.a2_in1k)|288     |82.62|95.75|60.2       |19.1 |37.3 |904    |
|[resnetaa50d.sw_in12k_ft_in1k](https://huggingface.co/timm/resnetaa50d.sw_in12k_ft_in1k)|288     |82.61|96.49|25.6       |8.9  |20.6 |1729   |
|[resnet61q.ra2_in1k](https://huggingface.co/timm/resnet61q.ra2_in1k)|288     |82.53|96.13|36.8       |9.9  |21.5 |1773   |
|[wide_resnet101_2.tv2_in1k](https://huggingface.co/timm/wide_resnet101_2.tv2_in1k)|224     |82.5 |96.02|126.9      |22.8 |21.2 |1078   |
|[resnext101_64x4d.c1_in1k](https://huggingface.co/timm/resnext101_64x4d.c1_in1k)|224     |82.46|95.92|83.5       |15.5 |31.2 |987    |
|[resnet51q.ra2_in1k](https://huggingface.co/timm/resnet51q.ra2_in1k)|288     |82.36|96.18|35.7       |8.1  |20.9 |1964   |
|[ecaresnet50t.ra2_in1k](https://huggingface.co/timm/ecaresnet50t.ra2_in1k)|320     |82.35|96.14|25.6       |8.8  |24.1 |1386   |
|[resnet101.a1_in1k](https://huggingface.co/timm/resnet101.a1_in1k)|288     |82.31|95.63|44.6       |13.0 |26.8 |1291   |
|[resnetrs101.tf_in1k](https://huggingface.co/timm/resnetrs101.tf_in1k)|288     |82.29|96.01|63.6       |13.6 |28.5 |1078   |
|[resnet152.tv2_in1k](https://huggingface.co/timm/resnet152.tv2_in1k)|224     |82.29|96.0 |60.2       |11.6 |22.6 |1484   |
|[wide_resnet50_2.racm_in1k](https://huggingface.co/timm/wide_resnet50_2.racm_in1k)|288     |82.27|96.06|68.9       |18.9 |23.8 |1176   |
|[resnet101d.ra2_in1k](https://huggingface.co/timm/resnet101d.ra2_in1k)|256     |82.26|96.07|44.6       |10.6 |22.2 |1542   |
|[resnet101.a2_in1k](https://huggingface.co/timm/resnet101.a2_in1k)|288     |82.24|95.73|44.6       |13.0 |26.8 |1290   |
|[seresnext50_32x4d.racm_in1k](https://huggingface.co/timm/seresnext50_32x4d.racm_in1k)|288     |82.2 |96.14|27.6       |7.0  |23.8 |1547   |
|[ecaresnet101d.miil_in1k](https://huggingface.co/timm/ecaresnet101d.miil_in1k)|224     |82.18|96.05|44.6       |8.1  |17.1 |1771   |
|[resnext50_32x4d.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnext50_32x4d.fb_swsl_ig1b_ft_in1k)|224     |82.17|96.22|25.0       |4.3  |14.4 |2943   |
|[ecaresnet50t.a1_in1k](https://huggingface.co/timm/ecaresnet50t.a1_in1k)|288     |82.12|95.65|25.6       |7.1  |19.6 |1704   |
|[resnext50_32x4d.a1h_in1k](https://huggingface.co/timm/resnext50_32x4d.a1h_in1k)|288     |82.03|95.94|25.0       |7.0  |23.8 |1745   |
|[ecaresnet101d_pruned.miil_in1k](https://huggingface.co/timm/ecaresnet101d_pruned.miil_in1k)|288     |82.0 |96.15|24.9       |5.8  |12.7 |1787   |
|[resnet61q.ra2_in1k](https://huggingface.co/timm/resnet61q.ra2_in1k)|256     |81.99|95.85|36.8       |7.8  |17.0 |2230   |
|[resnext101_32x8d.tv2_in1k](https://huggingface.co/timm/resnext101_32x8d.tv2_in1k)|176     |81.98|95.72|88.8       |10.3 |19.4 |1768   |
|[resnet152.a1_in1k](https://huggingface.co/timm/resnet152.a1_in1k)|224     |81.97|95.24|60.2       |11.6 |22.6 |1486   |
|[resnet101.a1h_in1k](https://huggingface.co/timm/resnet101.a1h_in1k)|224     |81.93|95.75|44.6       |7.8  |16.2 |2122   |
|[resnet101.tv2_in1k](https://huggingface.co/timm/resnet101.tv2_in1k)|224     |81.9 |95.77|44.6       |7.8  |16.2 |2118   |
|[resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/timm/resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k)|224     |81.84|96.1 |194.0      |36.3 |51.2 |583    |
|[resnet51q.ra2_in1k](https://huggingface.co/timm/resnet51q.ra2_in1k)|256     |81.78|95.94|35.7       |6.4  |16.6 |2471   |
|[resnet152.a2_in1k](https://huggingface.co/timm/resnet152.a2_in1k)|224     |81.77|95.22|60.2       |11.6 |22.6 |1485   |
|[resnetaa50d.sw_in12k_ft_in1k](https://huggingface.co/timm/resnetaa50d.sw_in12k_ft_in1k)|224     |81.74|96.06|25.6       |5.4  |12.4 |2813   |
|[ecaresnet50t.a2_in1k](https://huggingface.co/timm/ecaresnet50t.a2_in1k)|288     |81.65|95.54|25.6       |7.1  |19.6 |1703   |
|[ecaresnet50d.miil_in1k](https://huggingface.co/timm/ecaresnet50d.miil_in1k)|288     |81.64|95.88|25.6       |7.2  |19.7 |1694   |
|[resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/timm/resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k)|224     |81.62|96.04|88.8       |16.5 |31.2 |1101   |
|[wide_resnet50_2.tv2_in1k](https://huggingface.co/timm/wide_resnet50_2.tv2_in1k)|224     |81.61|95.76|68.9       |11.4 |14.4 |1930   |
|[resnetaa50.a1h_in1k](https://huggingface.co/timm/resnetaa50.a1h_in1k)|288     |81.61|95.83|25.6       |8.5  |19.2 |1868   |
|[resnet101.a1_in1k](https://huggingface.co/timm/resnet101.a1_in1k)|224     |81.5 |95.16|44.6       |7.8  |16.2 |2125   |
|[resnext50_32x4d.a1_in1k](https://huggingface.co/timm/resnext50_32x4d.a1_in1k)|288     |81.48|95.16|25.0       |7.0  |23.8 |1745   |
|[gcresnet50t.ra2_in1k](https://huggingface.co/timm/gcresnet50t.ra2_in1k)|288     |81.47|95.71|25.9       |6.9  |18.6 |2071   |
|[wide_resnet50_2.racm_in1k](https://huggingface.co/timm/wide_resnet50_2.racm_in1k)|224     |81.45|95.53|68.9       |11.4 |14.4 |1929   |
|[resnet50d.a1_in1k](https://huggingface.co/timm/resnet50d.a1_in1k)|288     |81.44|95.22|25.6       |7.2  |19.7 |1908   |
|[ecaresnet50t.ra2_in1k](https://huggingface.co/timm/ecaresnet50t.ra2_in1k)|256     |81.44|95.67|25.6       |5.6  |15.4 |2168   |
|[ecaresnetlight.miil_in1k](https://huggingface.co/timm/ecaresnetlight.miil_in1k)|288     |81.4 |95.82|30.2       |6.8  |13.9 |2132   |
|[resnet50d.ra2_in1k](https://huggingface.co/timm/resnet50d.ra2_in1k)|288     |81.37|95.74|25.6       |7.2  |19.7 |1910   |
|[resnet101.a2_in1k](https://huggingface.co/timm/resnet101.a2_in1k)|224     |81.32|95.19|44.6       |7.8  |16.2 |2125   |
|[seresnet50.ra2_in1k](https://huggingface.co/timm/seresnet50.ra2_in1k)|288     |81.3 |95.65|28.1       |6.8  |18.4 |1803   |
|[resnext50_32x4d.a2_in1k](https://huggingface.co/timm/resnext50_32x4d.a2_in1k)|288     |81.3 |95.11|25.0       |7.0  |23.8 |1746   |
|[seresnext50_32x4d.racm_in1k](https://huggingface.co/timm/seresnext50_32x4d.racm_in1k)|224     |81.27|95.62|27.6       |4.3  |14.4 |2591   |
|[ecaresnet50t.a1_in1k](https://huggingface.co/timm/ecaresnet50t.a1_in1k)|224     |81.26|95.16|25.6       |4.3  |11.8 |2823   |
|[gcresnext50ts.ch_in1k](https://huggingface.co/timm/gcresnext50ts.ch_in1k)|288     |81.23|95.54|15.7       |4.8  |19.6 |2117   |
|[senet154.gluon_in1k](https://huggingface.co/timm/senet154.gluon_in1k)|224     |81.23|95.35|115.1      |20.8 |38.7 |545    |
|[resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k)|288     |81.22|95.11|25.6       |6.8  |18.4 |2089   |
|[resnet50_gn.a1h_in1k](https://huggingface.co/timm/resnet50_gn.a1h_in1k)|288     |81.22|95.63|25.6       |6.8  |18.4 |676    |
|[resnet50d.a2_in1k](https://huggingface.co/timm/resnet50d.a2_in1k)|288     |81.18|95.09|25.6       |7.2  |19.7 |1908   |
|[resnet50.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnet50.fb_swsl_ig1b_ft_in1k)|224     |81.18|95.98|25.6       |4.1  |11.1 |3455   |
|[resnext50_32x4d.tv2_in1k](https://huggingface.co/timm/resnext50_32x4d.tv2_in1k)|224     |81.17|95.34|25.0       |4.3  |14.4 |2933   |
|[resnext50_32x4d.a1h_in1k](https://huggingface.co/timm/resnext50_32x4d.a1h_in1k)|224     |81.1 |95.33|25.0       |4.3  |14.4 |2934   |
|[seresnet50.a2_in1k](https://huggingface.co/timm/seresnet50.a2_in1k)|288     |81.1 |95.23|28.1       |6.8  |18.4 |1801   |
|[seresnet50.a1_in1k](https://huggingface.co/timm/seresnet50.a1_in1k)|288     |81.1 |95.12|28.1       |6.8  |18.4 |1799   |
|[resnet152s.gluon_in1k](https://huggingface.co/timm/resnet152s.gluon_in1k)|224     |81.02|95.41|60.3       |12.9 |25.0 |1347   |
|[resnet50.d_in1k](https://huggingface.co/timm/resnet50.d_in1k)|288     |80.97|95.44|25.6       |6.8  |18.4 |2085   |
|[gcresnet50t.ra2_in1k](https://huggingface.co/timm/gcresnet50t.ra2_in1k)|256     |80.94|95.45|25.9       |5.4  |14.7 |2571   |
|[resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/timm/resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k)|224     |80.93|95.73|44.2       |8.0  |21.2 |1814   |
|[resnet50.c1_in1k](https://huggingface.co/timm/resnet50.c1_in1k)|288     |80.91|95.55|25.6       |6.8  |18.4 |2084   |
|[seresnext101_32x4d.gluon_in1k](https://huggingface.co/timm/seresnext101_32x4d.gluon_in1k)|224     |80.9 |95.31|49.0       |8.0  |21.3 |1585   |
|[seresnext101_64x4d.gluon_in1k](https://huggingface.co/timm/seresnext101_64x4d.gluon_in1k)|224     |80.9 |95.3 |88.2       |15.5 |31.2 |918    |
|[resnet50.c2_in1k](https://huggingface.co/timm/resnet50.c2_in1k)|288     |80.86|95.52|25.6       |6.8  |18.4 |2085   |
|[resnet50.tv2_in1k](https://huggingface.co/timm/resnet50.tv2_in1k)|224     |80.85|95.43|25.6       |4.1  |11.1 |3450   |
|[ecaresnet50t.a2_in1k](https://huggingface.co/timm/ecaresnet50t.a2_in1k)|224     |80.84|95.02|25.6       |4.3  |11.8 |2821   |
|[ecaresnet101d_pruned.miil_in1k](https://huggingface.co/timm/ecaresnet101d_pruned.miil_in1k)|224     |80.79|95.62|24.9       |3.5  |7.7  |2961   |
|[seresnet33ts.ra2_in1k](https://huggingface.co/timm/seresnet33ts.ra2_in1k)|288     |80.79|95.36|19.8       |6.0  |14.8 |2506   |
|[ecaresnet50d_pruned.miil_in1k](https://huggingface.co/timm/ecaresnet50d_pruned.miil_in1k)|288     |80.79|95.58|19.9       |4.2  |10.6 |2349   |
|[resnet50.a2_in1k](https://huggingface.co/timm/resnet50.a2_in1k)|288     |80.78|94.99|25.6       |6.8  |18.4 |2088   |
|[resnet50.b1k_in1k](https://huggingface.co/timm/resnet50.b1k_in1k)|288     |80.71|95.43|25.6       |6.8  |18.4 |2087   |
|[resnext50_32x4d.ra_in1k](https://huggingface.co/timm/resnext50_32x4d.ra_in1k)|288     |80.7 |95.39|25.0       |7.0  |23.8 |1749   |
|[resnetrs101.tf_in1k](https://huggingface.co/timm/resnetrs101.tf_in1k)|192     |80.69|95.24|63.6       |6.0  |12.7 |2270   |
|[resnet50d.a1_in1k](https://huggingface.co/timm/resnet50d.a1_in1k)|224     |80.68|94.71|25.6       |4.4  |11.9 |3162   |
|[eca_resnet33ts.ra2_in1k](https://huggingface.co/timm/eca_resnet33ts.ra2_in1k)|288     |80.68|95.36|19.7       |6.0  |14.8 |2637   |
|[resnet50.a1h_in1k](https://huggingface.co/timm/resnet50.a1h_in1k)|224     |80.67|95.3 |25.6       |4.1  |11.1 |3452   |
|[resnext50d_32x4d.bt_in1k](https://huggingface.co/timm/resnext50d_32x4d.bt_in1k)|288     |80.67|95.42|25.0       |7.4  |25.1 |1626   |
|[resnetaa50.a1h_in1k](https://huggingface.co/timm/resnetaa50.a1h_in1k)|224     |80.63|95.21|25.6       |5.2  |11.6 |3034   |
|[ecaresnet50d.miil_in1k](https://huggingface.co/timm/ecaresnet50d.miil_in1k)|224     |80.61|95.32|25.6       |4.4  |11.9 |2813   |
|[resnext101_64x4d.gluon_in1k](https://huggingface.co/timm/resnext101_64x4d.gluon_in1k)|224     |80.61|94.99|83.5       |15.5 |31.2 |989    |
|[gcresnet33ts.ra2_in1k](https://huggingface.co/timm/gcresnet33ts.ra2_in1k)|288     |80.6 |95.31|19.9       |6.0  |14.8 |2578   |
|[gcresnext50ts.ch_in1k](https://huggingface.co/timm/gcresnext50ts.ch_in1k)|256     |80.57|95.17|15.7       |3.8  |15.5 |2710   |
|[resnet152.a3_in1k](https://huggingface.co/timm/resnet152.a3_in1k)|224     |80.56|95.0 |60.2       |11.6 |22.6 |1483   |
|[resnet50d.ra2_in1k](https://huggingface.co/timm/resnet50d.ra2_in1k)|224     |80.53|95.16|25.6       |4.4  |11.9 |3164   |
|[resnext50_32x4d.a1_in1k](https://huggingface.co/timm/resnext50_32x4d.a1_in1k)|224     |80.53|94.46|25.0       |4.3  |14.4 |2930   |
|[wide_resnet101_2.tv2_in1k](https://huggingface.co/timm/wide_resnet101_2.tv2_in1k)|176     |80.48|94.98|126.9      |14.3 |13.2 |1719   |
|[resnet152d.gluon_in1k](https://huggingface.co/timm/resnet152d.gluon_in1k)|224     |80.47|95.2 |60.2       |11.8 |23.4 |1428   |
|[resnet50.b2k_in1k](https://huggingface.co/timm/resnet50.b2k_in1k)|288     |80.45|95.32|25.6       |6.8  |18.4 |2086   |
|[ecaresnetlight.miil_in1k](https://huggingface.co/timm/ecaresnetlight.miil_in1k)|224     |80.45|95.24|30.2       |4.1  |8.4  |3530   |
|[resnext50_32x4d.a2_in1k](https://huggingface.co/timm/resnext50_32x4d.a2_in1k)|224     |80.45|94.63|25.0       |4.3  |14.4 |2936   |
|[wide_resnet50_2.tv2_in1k](https://huggingface.co/timm/wide_resnet50_2.tv2_in1k)|176     |80.43|95.09|68.9       |7.3  |9.0  |3015   |
|[resnet101d.gluon_in1k](https://huggingface.co/timm/resnet101d.gluon_in1k)|224     |80.42|95.01|44.6       |8.1  |17.0 |2007   |
|[resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k)|224     |80.38|94.6 |25.6       |4.1  |11.1 |3461   |
|[seresnet33ts.ra2_in1k](https://huggingface.co/timm/seresnet33ts.ra2_in1k)|256     |80.36|95.1 |19.8       |4.8  |11.7 |3267   |
|[resnext101_32x4d.gluon_in1k](https://huggingface.co/timm/resnext101_32x4d.gluon_in1k)|224     |80.34|94.93|44.2       |8.0  |21.2 |1814   |
|[resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/timm/resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k)|224     |80.32|95.4 |25.0       |4.3  |14.4 |2941   |
|[resnet101s.gluon_in1k](https://huggingface.co/timm/resnet101s.gluon_in1k)|224     |80.28|95.16|44.7       |9.2  |18.6 |1851   |
|[seresnet50.ra2_in1k](https://huggingface.co/timm/seresnet50.ra2_in1k)|224     |80.26|95.08|28.1       |4.1  |11.1 |2972   |
|[resnetblur50.bt_in1k](https://huggingface.co/timm/resnetblur50.bt_in1k)|288     |80.24|95.24|25.6       |8.5  |19.9 |1523   |
|[resnet50d.a2_in1k](https://huggingface.co/timm/resnet50d.a2_in1k)|224     |80.22|94.63|25.6       |4.4  |11.9 |3162   |
|[resnet152.tv2_in1k](https://huggingface.co/timm/resnet152.tv2_in1k)|176     |80.2 |94.64|60.2       |7.2  |14.0 |2346   |
|[seresnet50.a2_in1k](https://huggingface.co/timm/seresnet50.a2_in1k)|224     |80.08|94.74|28.1       |4.1  |11.1 |2969   |
|[eca_resnet33ts.ra2_in1k](https://huggingface.co/timm/eca_resnet33ts.ra2_in1k)|256     |80.08|94.97|19.7       |4.8  |11.7 |3284   |
|[gcresnet33ts.ra2_in1k](https://huggingface.co/timm/gcresnet33ts.ra2_in1k)|256     |80.06|94.99|19.9       |4.8  |11.7 |3216   |
|[resnet50_gn.a1h_in1k](https://huggingface.co/timm/resnet50_gn.a1h_in1k)|224     |80.06|94.95|25.6       |4.1  |11.1 |1109   |
|[seresnet50.a1_in1k](https://huggingface.co/timm/seresnet50.a1_in1k)|224     |80.02|94.71|28.1       |4.1  |11.1 |2962   |
|[resnet50.ram_in1k](https://huggingface.co/timm/resnet50.ram_in1k)|288     |79.97|95.05|25.6       |6.8  |18.4 |2086   |
|[resnet152c.gluon_in1k](https://huggingface.co/timm/resnet152c.gluon_in1k)|224     |79.92|94.84|60.2       |11.8 |23.4 |1455   |
|[seresnext50_32x4d.gluon_in1k](https://huggingface.co/timm/seresnext50_32x4d.gluon_in1k)|224     |79.91|94.82|27.6       |4.3  |14.4 |2591   |
|[resnet50.d_in1k](https://huggingface.co/timm/resnet50.d_in1k)|224     |79.91|94.67|25.6       |4.1  |11.1 |3456   |
|[resnet101.tv2_in1k](https://huggingface.co/timm/resnet101.tv2_in1k)|176     |79.9 |94.6 |44.6       |4.9  |10.1 |3341   |
|[resnetrs50.tf_in1k](https://huggingface.co/timm/resnetrs50.tf_in1k)|224     |79.89|94.97|35.7       |4.5  |12.1 |2774   |
|[resnet50.c2_in1k](https://huggingface.co/timm/resnet50.c2_in1k)|224     |79.88|94.87|25.6       |4.1  |11.1 |3455   |
|[ecaresnet26t.ra2_in1k](https://huggingface.co/timm/ecaresnet26t.ra2_in1k)|320     |79.86|95.07|16.0       |5.2  |16.4 |2168   |
|[resnet50.a2_in1k](https://huggingface.co/timm/resnet50.a2_in1k)|224     |79.85|94.56|25.6       |4.1  |11.1 |3460   |
|[resnet50.ra_in1k](https://huggingface.co/timm/resnet50.ra_in1k)|288     |79.83|94.97|25.6       |6.8  |18.4 |2087   |
|[resnet101.a3_in1k](https://huggingface.co/timm/resnet101.a3_in1k)|224     |79.82|94.62|44.6       |7.8  |16.2 |2114   |
|[resnext50_32x4d.ra_in1k](https://huggingface.co/timm/resnext50_32x4d.ra_in1k)|224     |79.76|94.6 |25.0       |4.3  |14.4 |2943   |
|[resnet50.c1_in1k](https://huggingface.co/timm/resnet50.c1_in1k)|224     |79.74|94.95|25.6       |4.1  |11.1 |3455   |
|[ecaresnet50d_pruned.miil_in1k](https://huggingface.co/timm/ecaresnet50d_pruned.miil_in1k)|224     |79.74|94.87|19.9       |2.5  |6.4  |3929   |
|[resnet33ts.ra2_in1k](https://huggingface.co/timm/resnet33ts.ra2_in1k)|288     |79.71|94.83|19.7       |6.0  |14.8 |2710   |
|[resnet152.gluon_in1k](https://huggingface.co/timm/resnet152.gluon_in1k)|224     |79.68|94.74|60.2       |11.6 |22.6 |1486   |
|[resnext50d_32x4d.bt_in1k](https://huggingface.co/timm/resnext50d_32x4d.bt_in1k)|224     |79.67|94.87|25.0       |4.5  |15.2 |2729   |
|[resnet50.bt_in1k](https://huggingface.co/timm/resnet50.bt_in1k)|288     |79.63|94.91|25.6       |6.8  |18.4 |2086   |
|[ecaresnet50t.a3_in1k](https://huggingface.co/timm/ecaresnet50t.a3_in1k)|224     |79.56|94.72|25.6       |4.3  |11.8 |2805   |
|[resnet101c.gluon_in1k](https://huggingface.co/timm/resnet101c.gluon_in1k)|224     |79.53|94.58|44.6       |8.1  |17.0 |2062   |
|[resnet50.b1k_in1k](https://huggingface.co/timm/resnet50.b1k_in1k)|224     |79.52|94.61|25.6       |4.1  |11.1 |3459   |
|[resnet50.tv2_in1k](https://huggingface.co/timm/resnet50.tv2_in1k)|176     |79.42|94.64|25.6       |2.6  |6.9  |5397   |
|[resnet32ts.ra2_in1k](https://huggingface.co/timm/resnet32ts.ra2_in1k)|288     |79.4 |94.66|18.0       |5.9  |14.6 |2752   |
|[resnet50.b2k_in1k](https://huggingface.co/timm/resnet50.b2k_in1k)|224     |79.38|94.57|25.6       |4.1  |11.1 |3459   |
|[resnext50_32x4d.tv2_in1k](https://huggingface.co/timm/resnext50_32x4d.tv2_in1k)|176     |79.37|94.3 |25.0       |2.7  |9.0  |4577   |
|[resnext50_32x4d.gluon_in1k](https://huggingface.co/timm/resnext50_32x4d.gluon_in1k)|224     |79.36|94.43|25.0       |4.3  |14.4 |2942   |
|[resnext101_32x8d.tv_in1k](https://huggingface.co/timm/resnext101_32x8d.tv_in1k)|224     |79.31|94.52|88.8       |16.5 |31.2 |1100   |
|[resnet101.gluon_in1k](https://huggingface.co/timm/resnet101.gluon_in1k)|224     |79.31|94.53|44.6       |7.8  |16.2 |2125   |
|[resnetblur50.bt_in1k](https://huggingface.co/timm/resnetblur50.bt_in1k)|224     |79.31|94.63|25.6       |5.2  |12.0 |2524   |
|[resnet50.a1h_in1k](https://huggingface.co/timm/resnet50.a1h_in1k)|176     |79.27|94.49|25.6       |2.6  |6.9  |5404   |
|[resnext50_32x4d.a3_in1k](https://huggingface.co/timm/resnext50_32x4d.a3_in1k)|224     |79.25|94.31|25.0       |4.3  |14.4 |2931   |
|[resnet50.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/timm/resnet50.fb_ssl_yfcc100m_ft_in1k)|224     |79.22|94.84|25.6       |4.1  |11.1 |3451   |
|[resnet33ts.ra2_in1k](https://huggingface.co/timm/resnet33ts.ra2_in1k)|256     |79.21|94.56|19.7       |4.8  |11.7 |3392   |
|[resnet50d.gluon_in1k](https://huggingface.co/timm/resnet50d.gluon_in1k)|224     |79.07|94.48|25.6       |4.4  |11.9 |3162   |
|[resnet50.ram_in1k](https://huggingface.co/timm/resnet50.ram_in1k)|224     |79.03|94.38|25.6       |4.1  |11.1 |3453   |
|[resnet50.am_in1k](https://huggingface.co/timm/resnet50.am_in1k)|224     |79.01|94.39|25.6       |4.1  |11.1 |3461   |
|[resnet32ts.ra2_in1k](https://huggingface.co/timm/resnet32ts.ra2_in1k)|256     |79.01|94.37|18.0       |4.6  |11.6 |3440   |
|[ecaresnet26t.ra2_in1k](https://huggingface.co/timm/ecaresnet26t.ra2_in1k)|256     |78.9 |94.54|16.0       |3.4  |10.5 |3421   |
|[resnet152.a3_in1k](https://huggingface.co/timm/resnet152.a3_in1k)|160     |78.89|94.11|60.2       |5.9  |11.5 |2745   |
|[wide_resnet101_2.tv_in1k](https://huggingface.co/timm/wide_resnet101_2.tv_in1k)|224     |78.84|94.28|126.9      |22.8 |21.2 |1079   |
|[seresnext26d_32x4d.bt_in1k](https://huggingface.co/timm/seresnext26d_32x4d.bt_in1k)|288     |78.83|94.24|16.8       |4.5  |16.8 |2251   |
|[resnet50.ra_in1k](https://huggingface.co/timm/resnet50.ra_in1k)|224     |78.81|94.32|25.6       |4.1  |11.1 |3454   |
|[seresnext26t_32x4d.bt_in1k](https://huggingface.co/timm/seresnext26t_32x4d.bt_in1k)|288     |78.74|94.33|16.8       |4.5  |16.7 |2264   |
|[resnet50s.gluon_in1k](https://huggingface.co/timm/resnet50s.gluon_in1k)|224     |78.72|94.23|25.7       |5.5  |13.5 |2796   |
|[resnet50d.a3_in1k](https://huggingface.co/timm/resnet50d.a3_in1k)|224     |78.71|94.24|25.6       |4.4  |11.9 |3154   |
|[wide_resnet50_2.tv_in1k](https://huggingface.co/timm/wide_resnet50_2.tv_in1k)|224     |78.47|94.09|68.9       |11.4 |14.4 |1934   |
|[resnet50.bt_in1k](https://huggingface.co/timm/resnet50.bt_in1k)|224     |78.46|94.27|25.6       |4.1  |11.1 |3454   |
|[resnet34d.ra2_in1k](https://huggingface.co/timm/resnet34d.ra2_in1k)|288     |78.43|94.35|21.8       |6.5  |7.5  |3291   |
|[gcresnext26ts.ch_in1k](https://huggingface.co/timm/gcresnext26ts.ch_in1k)|288     |78.42|94.04|10.5       |3.1  |13.3 |3226   |
|[resnet26t.ra2_in1k](https://huggingface.co/timm/resnet26t.ra2_in1k)|320     |78.33|94.13|16.0       |5.2  |16.4 |2391   |
|[resnet152.tv_in1k](https://huggingface.co/timm/resnet152.tv_in1k)|224     |78.32|94.04|60.2       |11.6 |22.6 |1487   |
|[seresnext26ts.ch_in1k](https://huggingface.co/timm/seresnext26ts.ch_in1k)|288     |78.28|94.1 |10.4       |3.1  |13.3 |3062   |
|[bat_resnext26ts.ch_in1k](https://huggingface.co/timm/bat_resnext26ts.ch_in1k)|256     |78.25|94.1 |10.7       |2.5  |12.5 |3393   |
|[resnet50.a3_in1k](https://huggingface.co/timm/resnet50.a3_in1k)|224     |78.06|93.78|25.6       |4.1  |11.1 |3450   |
|[resnet50c.gluon_in1k](https://huggingface.co/timm/resnet50c.gluon_in1k)|224     |78.0 |93.99|25.6       |4.4  |11.9 |3286   |
|[eca_resnext26ts.ch_in1k](https://huggingface.co/timm/eca_resnext26ts.ch_in1k)|288     |78.0 |93.91|10.3       |3.1  |13.3 |3297   |
|[seresnext26t_32x4d.bt_in1k](https://huggingface.co/timm/seresnext26t_32x4d.bt_in1k)|224     |77.98|93.75|16.8       |2.7  |10.1 |3841   |
|[resnet34.a1_in1k](https://huggingface.co/timm/resnet34.a1_in1k)|288     |77.92|93.77|21.8       |6.1  |6.2  |3609   |
|[resnet101.a3_in1k](https://huggingface.co/timm/resnet101.a3_in1k)|160     |77.88|93.71|44.6       |4.0  |8.3  |3926   |
|[resnet26t.ra2_in1k](https://huggingface.co/timm/resnet26t.ra2_in1k)|256     |77.87|93.84|16.0       |3.4  |10.5 |3772   |
|[seresnext26ts.ch_in1k](https://huggingface.co/timm/seresnext26ts.ch_in1k)|256     |77.86|93.79|10.4       |2.4  |10.5 |4263   |
|[resnetrs50.tf_in1k](https://huggingface.co/timm/resnetrs50.tf_in1k)|160     |77.82|93.81|35.7       |2.3  |6.2  |5238   |
|[gcresnext26ts.ch_in1k](https://huggingface.co/timm/gcresnext26ts.ch_in1k)|256     |77.81|93.82|10.5       |2.4  |10.5 |4183   |
|[ecaresnet50t.a3_in1k](https://huggingface.co/timm/ecaresnet50t.a3_in1k)|160     |77.79|93.6 |25.6       |2.2  |6.0  |5329   |
|[resnext50_32x4d.a3_in1k](https://huggingface.co/timm/resnext50_32x4d.a3_in1k)|160     |77.73|93.32|25.0       |2.2  |7.4  |5576   |
|[resnext50_32x4d.tv_in1k](https://huggingface.co/timm/resnext50_32x4d.tv_in1k)|224     |77.61|93.7 |25.0       |4.3  |14.4 |2944   |
|[seresnext26d_32x4d.bt_in1k](https://huggingface.co/timm/seresnext26d_32x4d.bt_in1k)|224     |77.59|93.61|16.8       |2.7  |10.2 |3807   |
|[resnet50.gluon_in1k](https://huggingface.co/timm/resnet50.gluon_in1k)|224     |77.58|93.72|25.6       |4.1  |11.1 |3455   |
|[eca_resnext26ts.ch_in1k](https://huggingface.co/timm/eca_resnext26ts.ch_in1k)|256     |77.44|93.56|10.3       |2.4  |10.5 |4284   |
|[resnet26d.bt_in1k](https://huggingface.co/timm/resnet26d.bt_in1k)|288     |77.41|93.63|16.0       |4.3  |13.5 |2907   |
|[resnet101.tv_in1k](https://huggingface.co/timm/resnet101.tv_in1k)|224     |77.38|93.54|44.6       |7.8  |16.2 |2125   |
|[resnet50d.a3_in1k](https://huggingface.co/timm/resnet50d.a3_in1k)|160     |77.22|93.27|25.6       |2.2  |6.1  |5982   |
|[resnext26ts.ra2_in1k](https://huggingface.co/timm/resnext26ts.ra2_in1k)|288     |77.17|93.47|10.3       |3.1  |13.3 |3392   |
|[resnet34.a2_in1k](https://huggingface.co/timm/resnet34.a2_in1k)|288     |77.15|93.27|21.8       |6.1  |6.2  |3615   |
|[resnet34d.ra2_in1k](https://huggingface.co/timm/resnet34d.ra2_in1k)|224     |77.1 |93.37|21.8       |3.9  |4.5  |5436   |
|[seresnet50.a3_in1k](https://huggingface.co/timm/seresnet50.a3_in1k)|224     |77.02|93.07|28.1       |4.1  |11.1 |2952   |
|[resnext26ts.ra2_in1k](https://huggingface.co/timm/resnext26ts.ra2_in1k)|256     |76.78|93.13|10.3       |2.4  |10.5 |4410   |
|[resnet26d.bt_in1k](https://huggingface.co/timm/resnet26d.bt_in1k)|224     |76.7 |93.17|16.0       |2.6  |8.2  |4859   |
|[resnet34.bt_in1k](https://huggingface.co/timm/resnet34.bt_in1k)|288     |76.5 |93.35|21.8       |6.1  |6.2  |3617   |
|[resnet34.a1_in1k](https://huggingface.co/timm/resnet34.a1_in1k)|224     |76.42|92.87|21.8       |3.7  |3.7  |5984   |
|[resnet26.bt_in1k](https://huggingface.co/timm/resnet26.bt_in1k)|288     |76.35|93.18|16.0       |3.9  |12.2 |3331   |
|[resnet50.tv_in1k](https://huggingface.co/timm/resnet50.tv_in1k)|224     |76.13|92.86|25.6       |4.1  |11.1 |3457   |
|[resnet50.a3_in1k](https://huggingface.co/timm/resnet50.a3_in1k)|160     |75.96|92.5 |25.6       |2.1  |5.7  |6490   |
|[resnet34.a2_in1k](https://huggingface.co/timm/resnet34.a2_in1k)|224     |75.52|92.44|21.8       |3.7  |3.7  |5991   |
|[resnet26.bt_in1k](https://huggingface.co/timm/resnet26.bt_in1k)|224     |75.3 |92.58|16.0       |2.4  |7.4  |5583   |
|[resnet34.bt_in1k](https://huggingface.co/timm/resnet34.bt_in1k)|224     |75.16|92.18|21.8       |3.7  |3.7  |5994   |
|[seresnet50.a3_in1k](https://huggingface.co/timm/seresnet50.a3_in1k)|160     |75.1 |92.08|28.1       |2.1  |5.7  |5513   |
|[resnet34.gluon_in1k](https://huggingface.co/timm/resnet34.gluon_in1k)|224     |74.57|91.98|21.8       |3.7  |3.7  |5984   |
|[resnet18d.ra2_in1k](https://huggingface.co/timm/resnet18d.ra2_in1k)|288     |73.81|91.83|11.7       |3.4  |5.4  |5196   |
|[resnet34.tv_in1k](https://huggingface.co/timm/resnet34.tv_in1k)|224     |73.32|91.42|21.8       |3.7  |3.7  |5979   |
|[resnet18.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k)|224     |73.28|91.73|11.7       |1.8  |2.5  |10213  |
|[resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k)|288     |73.16|91.03|11.7       |3.0  |4.1  |6050   |
|[resnet34.a3_in1k](https://huggingface.co/timm/resnet34.a3_in1k)|224     |72.98|91.11|21.8       |3.7  |3.7  |5967   |
|[resnet18.fb_ssl_yfcc100m_ft_in1k](https://huggingface.co/timm/resnet18.fb_ssl_yfcc100m_ft_in1k)|224     |72.6 |91.42|11.7       |1.8  |2.5  |10213  |
|[resnet18.a2_in1k](https://huggingface.co/timm/resnet18.a2_in1k)|288     |72.37|90.59|11.7       |3.0  |4.1  |6051   |
|[resnet14t.c3_in1k](https://huggingface.co/timm/resnet14t.c3_in1k)|224     |72.26|90.31|10.1       |1.7  |5.8  |7026   |
|[resnet18d.ra2_in1k](https://huggingface.co/timm/resnet18d.ra2_in1k)|224     |72.26|90.68|11.7       |2.1  |3.3  |8707   |
|[resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k)|224     |71.49|90.07|11.7       |1.8  |2.5  |10187  |
|[resnet14t.c3_in1k](https://huggingface.co/timm/resnet14t.c3_in1k)|176     |71.31|89.69|10.1       |1.1  |3.6  |10970  |
|[resnet18.gluon_in1k](https://huggingface.co/timm/resnet18.gluon_in1k)|224     |70.84|89.76|11.7       |1.8  |2.5  |10210  |
|[resnet18.a2_in1k](https://huggingface.co/timm/resnet18.a2_in1k)|224     |70.64|89.47|11.7       |1.8  |2.5  |10194  |
|[resnet34.a3_in1k](https://huggingface.co/timm/resnet34.a3_in1k)|160     |70.56|89.52|21.8       |1.9  |1.9  |10737  |
|[resnet18.tv_in1k](https://huggingface.co/timm/resnet18.tv_in1k)|224     |69.76|89.07|11.7       |1.8  |2.5  |10205  |
|[resnet10t.c3_in1k](https://huggingface.co/timm/resnet10t.c3_in1k)|224     |68.34|88.03|5.4        |1.1  |2.4  |13079  |
|[resnet18.a3_in1k](https://huggingface.co/timm/resnet18.a3_in1k)|224     |68.25|88.17|11.7       |1.8  |2.5  |10167  |
|[resnet10t.c3_in1k](https://huggingface.co/timm/resnet10t.c3_in1k)|176     |66.71|86.96|5.4        |0.7  |1.5  |20327  |
|[resnet18.a3_in1k](https://huggingface.co/timm/resnet18.a3_in1k)|160     |65.66|86.26|11.7       |0.9  |1.3  |18229  |

## Citation
```bibtex
@misc{yalniz2019billionscale,
    title={Billion-scale semi-supervised learning for image classification},
    author={I. Zeki Yalniz and Hervé Jégou and Kan Chen and Manohar Paluri and Dhruv Mahajan},
    year={2019},
    eprint={1905.00546},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```bibtex
@article{Xie2016,
  title={Aggregated Residual Transformations for Deep Neural Networks},
  author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
```
```bibtex
@article{He2015,
  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title = {Deep Residual Learning for Image Recognition},
  journal = {arXiv preprint arXiv:1512.03385},
  year = {2015}
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
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
