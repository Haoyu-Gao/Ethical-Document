---
language:
- en
license: openrail++
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- stable-diffusion
- text-to-image
- diffusers
- ffai
base_model: FFusion/FFusionXL-BASE
inference: true
widget:
- text: a dog in colorful exploding clouds, dreamlike surrealism colorful smoke and
    fire coming out of it, explosion of data fragments, exploding background,realistic
    explosion, 3d digital art
  example_title: Dogo FFusion
- text: a sprinkled donut sitting on top of a table, colorful hyperrealism, everything
    is made of candy, hyperrealistic digital painting, covered in sprinkles and crumbs,
    vibrant colors hyper realism,colorful smoke explosion background
  example_title: Donut FFusion
- text: a cup of coffee with a tree in it, surreal art, awesome great composition,
    surrealism, ice cubes in tree, colorful clouds, perfectly realistic yet surreal
  example_title: CoFFee FFusion
- text: brightly colored headphones with a splash of colorful paint splash, vibing
    to music, stunning artwork, music is life, beautiful digital artwork, concept
    art, cinematic, dramatic, intricate details, dark lighting
  example_title: Headset FFusion
- text: high-quality game character digital design, Unreal Engine, Water color painting,
    Mecha- Monstrous high quality game fantasy rpg character design, dark rainbow
    Fur Scarf, inside of a Superficial Outhouse, at Twilight, Overdetailed art
  example_title: Digital Fusion
thumbnail: https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image7sm.jpg
---

# FFXL400 Combined LoRA Model 🚀

Welcome to the FFXL400 combined LoRA model repository on Hugging Face! This model is a culmination of extensive research, bringing together the finest LoRAs from the [400GB-LoraXL repository](https://huggingface.co/FFusion/400GB-LoraXL). Our vision was to harness the power of multiple LoRAs, meticulously analyzing and integrating a select fraction of the blocks from each. 

## 📦 Model Highlights

- **Innovative Combination**: This model is a strategic integration of LoRAs, maximizing the potential of each while creating a unified powerhouse.
- **Versatility**: The model is available in various formats including diffusers, safetensors (both fp 16 and 32), and an optimized ONNIX FP16 version for DirectML, ensuring compatibility across AMD, Intel, Nvidia, and more.
- **Advanced Research**: Leveraging the latest in machine learning research, the model represents a state-of-the-art amalgamation of LoRAs, optimized for performance and accuracy.

## 🔍 Technical Insights

This model is a testament to the advancements in the field of AI and machine learning. It was crafted with precision, ensuring that:

- Only a small percentage of the blocks from the original LoRAs (UNet and text encoders) were utilized.
- The model is primed not just for inference but also for further training and refinement.
- It serves as a benchmark for testing and understanding the cumulative impact of multiple LoRAs when used in concert.

## 🎨 Usage 

The FFXL400 model is designed for a multitude of applications. Whether you're delving into research, embarking on a new project, or simply experimenting, this model serves as a robust foundation. Use it to:

- Investigate the cumulative effects of merging multiple LoRAs.
- Dive deep into weighting experiments with multiple LoRAs.
- Explore the nuances and intricacies of integrated LoRAs.



## ⚠️ License & Usage Disclaimers

**Please review the full [license agreement](https://huggingface.co/FFusion/FFXL400/blob/main/LICENSE.md) before accessing or using the models.**

🔴 The models and weights available in this repository are **strictly for research and testing purposes**, with exceptions noted below. They are **not** generally intended for commercial use and are dependent on each individual LORA. 

🔵 **Exception for Commercial Use:** The [FFusionXL-BASE](https://huggingface.co/FFusion/FFusionXL-BASE), [FFusion-BaSE](https://huggingface.co/FFusion/FFusion-BaSE), [di.FFUSION.ai-v2.1-768-BaSE-alpha](https://huggingface.co/FFusion/di.FFUSION.ai-v2.1-768-BaSE-alpha), and [di.ffusion.ai.Beta512](https://huggingface.co/FFusion/di.ffusion.ai.Beta512) models are trained by FFusion AI using images for which we hold licenses. Users are advised to primarily use these models for a safer experience. These particular models are allowed for commercial use.

🔴 **Disclaimer:** FFusion AI, in conjunction with Source Code Bulgaria Ltd and BlackswanTechnologies, **does not endorse or guarantee the content produced by the weights in each LORA**. There's potential for generating NSFW or offensive content. Collectively, we expressly disclaim responsibility for the outcomes and content produced by these weights.

🔴 **Acknowledgement:** The [FFusionXL-BASE](https://huggingface.co/FFusion/FFusionXL-BASE) model model is a uniquely developed version by FFusion AI. Rights to this and associated modifications belong to FFusion AI and Source Code Bulgaria Ltd. Ensure adherence to both this license and any conditions set by Stability AI Ltd for referenced models.


## 📈 How to Use

The model can be easily integrated into your projects. Here's a quick guide on how to use the FFXL400 model:

1. **Loading the Model**:
    ```python
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("FFusion/FFXL400")
    model = AutoModel.from_pretrained("FFusion/FFXL400")
    ```

2. **Performing Inference**:
    ```python
    input_text = "Your input here"
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    ```

## Further Training

    You can also use the FFXL400 as a starting point for further training. Simply load it into your training pipeline and proceed as you would with any other model.
	
[Autotrain Advanced](https://github.com/huggingface/autotrain-advanced),	

[Kohya + Stable Diffusion XL](https://huggingface.co/docs/diffusers/main/en/training/lora#stable-diffusion-xl),	



## 📚 Background

The FFXL400 is built upon the insights and data from the [400GB-LoraXL repository](https://huggingface.co/FFusion/400GB-LoraXL). Each LoRA in that collection was extracted using the Low-Rank Adaptation (LoRA) technique, providing a rich dataset for research and exploration. The FFXL400 is the pinnacle of that research, representing a harmonious blend of the best LoRAs.


## Library of Available LoRA Models 📚

![loraXL FFUsion](https://cdn-uploads.huggingface.co/production/uploads/6380cf05f496d57325c12194/XQlnis5W2-fgDnGZ60EK9.jpeg)

You can choose any of the models from our repository on Hugging Face or the upcoming repository on CivitAI. Here's a list of available models with `lora_model_id = "FFusion/400GB-LoraXL"`:

```
lora_filename = 
    - FFai.0001.4Guofeng4xl_V1125d.lora_Dim64.safetensors
    - FFai.0002.4Guofeng4xl_V1125d.lora_Dim8.safetensors
    - FFai.0003.4Guofeng4xl_V1125d.loraa.safetensors
    - FFai.0004.Ambiencesdxl_A1.lora.safetensors
    - FFai.0005.Ambiencesdxl_A1.lora_8.safetensors
    - FFai.0006.Angrasdxl10_V22.lora.safetensors
    - FFai.0007.Animaginexl_V10.lora.safetensors
    - FFai.0008.Animeartdiffusionxl_Alpha3.lora.safetensors
    - FFai.0009.Astreapixiexlanime_V16.lora.safetensors
    - FFai.0010.Bluepencilxl_V010.lora.safetensors
    - FFai.0011.Bluepencilxl_V021.lora.safetensors
    - FFai.0012.Breakdomainxl_V03d.lora.safetensors
    - FFai.0013.Canvasxl_Bfloat16v002.lora.safetensors
    - FFai.0014.Cherrypickerxl_V20.lora.safetensors
    - FFai.0015.Copaxtimelessxlsdxl1_V44.lora.safetensors
    - FFai.0016.Counterfeitxl-Ffusionai-Alpha-Vae.lora.safetensors
    - FFai.0017.Counterfeitxl_V10.lora.safetensors
    - FFai.0018.Crystalclearxl_Ccxl.lora.safetensors
    - FFai.0019.Deepbluexl_V006.lora.safetensors
    - FFai.0020.Dream-Ffusion-Shaper.lora.safetensors
    - FFai.0021.Dreamshaperxl10_Alpha2xl10.lora.safetensors
    - FFai.0022.Duchaitenaiartsdxl_V10.lora.safetensors
    - FFai.0023.Dynavisionxlallinonestylized_Beta0371bakedvae.lora.safetensors
    - FFai.0024.Dynavisionxlallinonestylized_Beta0411bakedvae.lora.safetensors
    - FFai.0025.Fantasticcharacters_V55.lora.safetensors
    - FFai.0026.Fenrisxl_V55.lora.safetensors
    - FFai.0027.Fudukimix_V10.lora.safetensors
    - FFai.0028.Infinianimexl_V16.lora.safetensors
    - FFai.0029.Juggernautxl_Version1.lora_1.safetensors
    - FFai.0030.Lahmysterioussdxl_V330.lora.safetensors
    - FFai.0031.Mbbxlultimate_V10rc.lora.safetensors
    - FFai.0032.Miamodelsfwnsfwsdxl_V30.lora.safetensors
    - FFai.0033.Morphxl_V10.lora.safetensors
    - FFai.0034.Nightvisionxlphotorealisticportrait_Beta0681bakedvae.lora_1.safetensors
    - FFai.0035.Osorubeshialphaxl_Z.lora.safetensors
    - FFai.0036.Physiogenxl_V04.lora.safetensors
    - FFai.0037.Protovisionxlhighfidelity3d_Beta0520bakedvae.lora.safetensors
    - FFai.0038.Realitycheckxl_Alpha11.lora.safetensors
    - FFai.0039.Realmixxl_V10.lora.safetensors
    - FFai.0040.Reproductionsdxl_V31.lora.safetensors
    - FFai.0041.Rundiffusionxl_Beta.lora.safetensors
    - FFai.0042.Samaritan3dcartoon_V40sdxl.lora.safetensors
    - FFai.0043.Sdvn6realxl_Detailface.lora.safetensors
    - FFai.0044.Sdvn7realartxl_Beta2.lora.safetensors
    - FFai.0045.Sdxl10arienmixxlasian_V10.lora.safetensors
    - FFai.0046.Sdxlbasensfwfaces_Sdxlnsfwfaces03.lora.safetensors
    - FFai.0047.Sdxlfaetastic_V10.lora.safetensors
    - FFai.0048.Sdxlfixedvaefp16remove_Basefxiedvaev2fp16.lora.safetensors
    - FFai.0049.Sdxlnijiv4_Sdxlnijiv4.lora.safetensors
    - FFai.0050.Sdxlronghua_V11.lora.safetensors
    - FFai.0051.Sdxlunstablediffusers_V5unchainedslayer.lora.safetensors
    - FFai.0052.Sdxlyamersanimeultra_Yamersanimev2.lora.safetensors
    - FFai.0053.Shikianimexl_V10.lora.safetensors
    - FFai.0054.Spectrumblendx_V10.lora.safetensors
    - FFai.0055.Stablediffusionxl_V30.lora.safetensors
    - FFai.0056.Talmendoxlsdxl_V11beta.lora.safetensors
    - FFai.0057.Wizard_V10.lora.safetensors
    - FFai.0058.Wyvernmix15xl_Xlv11.lora.safetensors
    - FFai.0059.Xl13asmodeussfwnsfw_V17bakedvae.lora.safetensors
    - FFai.0060.Xl3experimentalsd10xl_V10.lora.safetensors
    - FFai.0061.Xl6hephaistossd10xlsfw_V21bakedvaefp16fix.lora.safetensors
    - FFai.0062.Xlperfectdesign_V2ultimateartwork.lora.safetensors
    - FFai.0063.Xlyamersrealistic_V3.lora.safetensors
    - FFai.0064.Xxmix9realisticsdxl_Testv20.lora.safetensors
    - FFai.0065.Zavychromaxl_B2.lora.safetensors

```

## 🎉 Acknowledgements & Citations

A huge shoutout to the community for their continued support and feedback. Together, we are pushing the boundaries of what's possible with machine learning!

We would also like to acknowledge and give credit to the following projects and authors:

- **ComfyUI**: We've used and modified portions of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for our work.
- **kohya-ss/sd-scripts and bmaltais**: Our work also incorporates modifications from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).
- **lora-inspector**: We've benefited from the [lora-inspector](https://github.com/rockerBOO/lora-inspector) project.
- **KohakuBlueleaf**: Special mention to KohakuBlueleaf for their invaluable contributions.


[![400GB FFusion Lora XL 5](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image5sm.jpg)](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image5.jpg)

[![400GB FFusion Lora XL 6](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image6sm.jpg)](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image6.jpg)

[![400GB FFusion Lora XL 7](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image7sm.jpg)](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image7.jpg)

[![400GB FFusion Lora XL 9](https://huggingface.co/FFusion/400GB-LoraXL/resolve/main/images/image9.jpg)](https://huggingface.co/FFusion/400GB-LoraXL/tree/main)


### HowMuch ???
![60% Works](https://img.shields.io/badge/60%25%20of%20the%20Time-It%20Works%20Every%20Time-green)


**Have you ever asked yourself, "How much space have I wasted on `*.ckpt` and `*.safetensors` checkpoints?"** 🤔
Say hello to HowMuch: Checking checkpoint wasted space since... well, now! 

😄 Enjoy this somewhat unnecessary, yet **"fun-for-the-whole-family"** DiskSpaceAnalyzer tool. 😄

## Overview


`HowMuch` is a Python tool designed to scan your drives (or a specified directory) and report on the total space used by files with specific extensions, mainly `.ckpt` and `.safetensors`. 

It outputs:
- The total storage capacity of each scanned drive or directory.
- The space occupied by `.ckpt` and `.safetensors` files.
- The free space available.
- A neat bar chart visualizing the above data.

## Installation
[GitHub](https://github.com/1e-2/HowMuch)

### From PyPI

You can easily install `HowMuch` via pip:

```bash
pip install howmuch
```

### From Source

1. Clone the repository:

   ```bash
   git clone https://github.com/1e-2/HowMuch.git
   ```

2. Navigate to the cloned directory and install:

   ```bash
   cd HowMuch
   pip install .
   ```

## Usage


Run the tool without any arguments to scan all drives:

```bash
howmuch
```

Or, specify a particular directory or drive to scan:

```bash
howmuch --scan C:
```

###  🌐 **Contact Information**

The **FFusion.ai** project is proudly maintained by **Source Code Bulgaria Ltd** & **Black Swan Technologies**.

📧 Reach us at [di@ffusion.ai](mailto:di@ffusion.ai) for any inquiries or support.

#### 🌌 **Find us on:** 

- 🐙 [GitHub](https://github.com/1e-2)
- 😊 [Hugging Face](https://huggingface.co/FFusion/)
- 💡 [Civitai](https://civitai.com/user/idle/models)

🔐 **Security powered by** [Comodo.BG](http://Comodo.BG) & [Preasidium.CX](http://Preasidium.CX)
🚀 Marketing by [Гугъл.com](http://Гугъл.com)
📩 [![Email](https://img.shields.io/badge/Email-enquiries%40ffusion.ai-blue?style=for-the-badge&logo=gmail)](mailto:enquiries@ffusion.ai)
🌍 Sofia Istanbul London


---

We hope the FFXL400 serves as a valuable asset in your AI journey. We encourage feedback, contributions, and insights from the community to further refine and enhance this model. Together, let's push the boundaries of what's possible!



![ffusionai-logo.png](https://cdn-uploads.huggingface.co/production/uploads/6380cf05f496d57325c12194/EjDa_uGcOoH2cXM2K-NYn.png)
