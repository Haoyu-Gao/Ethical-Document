---
language:
- en
license: creativeml-openrail-m
library_name: diffusers
tags:
- stable-diffusion
- stable-diffusion-diffusers
pipeline_tag: text-to-image
duplicated_from: xiaolxl/Gf_style2
---
# Gf_style2 - 介绍

欢迎使用Gf_style2模型 - 这是一个中国华丽古风风格模型，也可以说是一个古风游戏角色模型，具有2.5D的质感。第二代相对与第一代减少了上手难度，不需要固定的配置也能生成好看的图片。同时也改进了上一代脸崩坏的问题。

这是一个模型系列，会在未来不断更新模型。

--

Welcome to Gf_ Style2 model - This is a Chinese gorgeous antique style model, which can also be said to be an antique game role model with a 2.5D texture. Compared with the first generation, the second generation reduces the difficulty of getting started and can generate beautiful pictures without fixed configuration. At the same time, it also improved the problem of face collapse of the previous generation.

This is a series of models that will be updated in the future.

3.0版本已发布：[https://huggingface.co/xiaolxl/Gf_style3](https://huggingface.co/xiaolxl/Gf_style3)

# install - 安装教程

1. 将XXX.ckpt模型放入SD目录 - Put XXX.ckpt model into SD directory

2. 模型自带VAE如果你的程序无法加载请记住选择任意一个VAE文件，否则图形将为灰色 - The model comes with VAE. If your program cannot be loaded, please remember to select any VAE file, otherwise the drawing will be gray

# How to use - 如何使用

(TIP:人物是竖图炼制，理论上生成竖图效果更好)

简单：第二代上手更加简单，你只需要下方3个设置即可 - simple：The second generation is easier to use. You only need the following three settings:

- The size of the picture should be at least **768**, otherwise it will collapse - 图片大小至少768，不然会崩图

- **key word(Start):**
```
{best quality}, {{masterpiece}}, {highres}, {an extremely delicate and beautiful}, original, extremely detailed wallpaper,1girl
```

- **Negative words - 感谢群友提供的负面词:**
```
(((simple background))),monochrome ,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, lowres, bad anatomy, bad hands, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mut ilated,tran nsexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,bad anatomy,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (((missing arms))),(((missing legs))), (((extra arms))),(((extra legs))),pubic hair, plump,bad legs,error legs,username,blurry,bad feet
```

高级：如果您还想使图片尽可能更好，请尝试以下配置 - senior：If you also want to make the picture as better as possible, please try the following configuration

- Sampling steps:**30 or 50**

- Sampler:**DPM++ SDE Karras**

- The size of the picture should be at least **768**, otherwise it will collapse - 图片大小至少768，不然会崩图

- If the face is deformed, try to Open **face repair**

- **如果想元素更丰富，可以添加下方关键词 - If you want to enrich the elements, you can add the following keywords**
```
strapless dress,
smile, 
china dress,dress,hair ornament, necklace, jewelry, long hair, earrings, chinese clothes,
```

# Examples - 例图

(可在文件列表中找到原图，并放入WebUi查看关键词等信息) - (You can find the original image in the file list, and put WebUi to view keywords and other information)

<img src=https://huggingface.co/xiaolxl/Gf_style2/resolve/main/examples/a1.png>

<img src=https://huggingface.co/xiaolxl/Gf_style2/resolve/main/examples/a2.png>