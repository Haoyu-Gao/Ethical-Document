---
{}
---
# ruclip-vit-base-patch32-384

**RuCLIP** (**Ru**ssian **C**ontrastive **L**anguage–**I**mage **P**retraining) is a multimodal model 
for obtaining images and text similarities and rearranging captions and pictures. 
RuCLIP builds on a large body of work on zero-shot transfer, computer vision, natural language processing and 
multimodal learning. 

Model was trained by [Sber AI](https://github.com/sberbank-ai) and [SberDevices](https://sberdevices.ru/) teams.  
* Task: `text ranking`; `image ranking`; `zero-shot image classification`;
* Type: `encoder`
* Num Parameters: `150M`
* Training Data Volume: `240 million text-image pairs`
* Language: `Russian`
* Context Length: `77`
* Transformer Layers: `12`
* Transformer Width: `512`
* Transformer Heads: `8`
* Image Size: `384`
* Vision Layers: `12`
* Vision Width: `768`
* Vision Patch Size: `32`

## Usage [Github](https://github.com/sberbank-ai/ru-clip)

```
pip install ruclip
```

```python
clip, processor = ruclip.load("ruclip-vit-base-patch32-384", device="cuda")
```


## Performance
We have evaluated the performance on the following datasets:

| Dataset       | Metric Name    | Metric Result               |
|:--------------|:---------------|:----------------------------|
| Food101       | acc            | 0.642                       |
| CIFAR10       | acc            | 0.862                       |
| CIFAR100      | acc            | 0.529                       |
| Birdsnap      | acc            | 0.161                       |
| SUN397        | acc            | 0.510                       |
| Stanford Cars | acc            | 0.572                       |
| DTD           | acc            | 0.390	                     |
| MNIST         | acc            | 0.404	                     |
| STL10         | acc            | 0.946	                     |
| PCam          | acc            | 0.506                       |
| CLEVR         | acc            | 0.188                       |
| Rendered SST2 | acc            | 0.508                       |
| ImageNet      | acc            | 0.451                       |
| FGVC Aircraft | mean-per-class | 0.053                       |
| Oxford Pets   | mean-per-class | 0.587                       |
| Caltech101    | mean-per-class | 0.834	                     |
| Flowers102    | mean-per-class | 0.449                       |
| HatefulMemes  | roc-auc        | 0.537                       |




# Authors

+ Alex Shonenkov: [Github](https://github.com/shonenkov), [Kaggle GM](https://www.kaggle.com/shonenkov)
+ Daniil Chesakov: [Github](https://github.com/Danyache)
+ Denis Dimitrov: [Github](https://github.com/denndimitrov)
+ Igor Pavlov: [Github](https://github.com/boomb0om)
