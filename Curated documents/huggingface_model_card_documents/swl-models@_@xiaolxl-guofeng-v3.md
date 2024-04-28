---
language:
- en
license: creativeml-openrail-m
library_name: diffusers
tags:
- stable-diffusion
- stable-diffusion-diffusers
pipeline_tag: text-to-image
duplicated_from: xiaolxl/GuoFeng3
---
# 介绍 - GuoFeng3

欢迎使用GuoFeng3模型 - (TIP:这个版本的名字进行了微调),这是一个中国华丽古风风格模型，也可以说是一个古风游戏角色模型，具有2.5D的质感。第三代大幅度减少上手难度，增加了场景元素与男性古风人物，除此之外为了模型能更好的适应其它TAG，还增加了其它风格的元素。这一代对脸和手的崩坏有一定的修复，同时素材大小也提高到了最长边1024。

--

Welcome to the GuoFeng3 model - (TIP: the name of this version has been fine-tuned). This is a Chinese gorgeous antique style model, which can also be said to be an antique game character model with a 2.5D texture. The third generation greatly reduces the difficulty of getting started, and adds scene elements and male antique characters. In addition, in order to better adapt the model to other TAGs, other style elements are also added. This generation has repaired the broken face and hands to a certain extent, and the size of the material has also increased to the longest side of 1024.

# 安装教程 - install

1. 将GuoFeng3.ckpt模型放入SD目录 - Put GuoFeng3.ckpt model into SD directory

2. 此模型自带VAE，如果你的程序不支持，请记得选择任意一个VAE文件，否则图形将为灰色 - This model comes with VAE. If your program does not support it, please remember to select any VAE file, otherwise the graphics will be gray

# 如何使用 - How to use

**TIP：经过一天的测试，发现很多人物可能出现红眼问题，可以尝试在负面词添加red eyes。如果色彩艳丽可以尝试降低CFG - After a day of testing, we found that many characters may have red-eye problems. We can try to add red eyes to negative words。Try to reduce CFG if the color is bright**

简单：第三代大幅度减少上手难度 - Simple: the third generation greatly reduces the difficulty of getting started

- **关键词 - key word:**
```
best quality, masterpiece, highres, 1girl,china dress,Beautiful face
```

- **负面词 - Negative words:**
```
NSFW, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet
```

---

高级：如果您还想使图片尽可能更好，请尝试以下配置 - senior：If you also want to make the picture as better as possible, please try the following configuration

- Sampling steps:**50**

- Sampler:**DPM++ SDE Karras or DDIM**

- The size of the picture should be at least **1024** - 图片大小至少1024

- CFG:**4-6**

- **更好的负面词 Better negative words - 感谢群友提供的负面词:**
```
(((simple background))),monochrome ,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, lowres, bad anatomy, bad hands, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mut ilated,tran nsexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,bad anatomy,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (((missing arms))),(((missing legs))), (((extra arms))),(((extra legs))),pubic hair, plump,bad legs,error legs,username,blurry,bad feet
```

- **如果想元素更丰富，可以添加下方关键词 - If you want to enrich the elements, you can add the following keywords**
```
Beautiful face,
hair ornament, solo,looking at viewer,smile,closed mouth,lips
china dress,dress,hair ornament, necklace, jewelry, long hair, earrings, chinese clothes,
architecture,east asian architecture,building,outdoors,rooftop,city,cityscape
```

# 例图 - Examples

(可在文件列表中找到原图，并放入WebUi查看关键词等信息) - (You can find the original image in the file list, and put WebUi to view keywords and other information)

<img src=https://huggingface.co/xiaolxl/GuoFeng3/resolve/main/examples/e1.png>

<img src=https://huggingface.co/xiaolxl/GuoFeng3/resolve/main/examples/e2.png>

<img src=https://huggingface.co/xiaolxl/GuoFeng3/resolve/main/examples/e3.png>

<img src=https://huggingface.co/xiaolxl/GuoFeng3/resolve/main/examples/e4.png>