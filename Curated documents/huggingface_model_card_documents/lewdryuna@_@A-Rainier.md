---
license: creativeml-openrail-m
tags:
- stable-diffusion
- text-to-image
duplicated_from: Hemlok/RainierMix
---

# ◆RainierMix

![a](Image/RainierMix.png)

- "RainierMix" is a merged model based on "ACertainThing".

---

# 《Notice》

- **"RainierMixV2" and "PastelRainier" are no longer available for commercial use due to a change in the license of the merging source.**

- Instead, we have created **"RainierMix-V2.5"** and **"PastelRainier-V2.5"** Please use them.

----

- The model was corrupted and a corrected version has been uploaded to "Modified-Model".

![a](Image/3.png)
- Here is an example of how it compares to the modified version.

---

# ◆Discord
[Join Discord Server](https://discord.gg/eN6aSWRddT)

- The merged model community of Hemlok.

----

# ◆About

- Sampler: DDIM or DPM++ SDE Karras

- Steps: 50~

- Clipskip: 2

- CFG Scale: 5-8

- Denoise strength: 0.5-0.7

- Negative prompts should be as few as possible.
- *Always use VAE to avoid possible color fading.*

----


# ◆Model Types
- Prompt:
```
kawaii, 1girl, (solo), (cowboy shot), (dynamic angle), Ruffled Dresses, (The great hall of the mansion), tiara, Luxurious interior, looking at viewer,
```

---

## ◇Rainier-base
![a](Image/base.png)
- ACertainThing + Anything-V4.5

---

## ◇RainierMixV1
![a](Image/V1.png)
- Rainier-base + Counterfeit-V2.0 + Evt_V4-preview

---

## ◇RainierMix-V2.5
![a](Image/v25.png)
- Neuauflage des Modells "RainierMixV2".


## ◇PastelRainier-V2.5
![a](Image/p25.png)
- Neuauflage des Modells "PastelRainier".

---

# ◆How to use

- Please download the file by yourself and use it with WebUI(AUTOMATIC1111) etc.

- Use the fp16 version for Colab(T4) or a PC with low RAM.

- The models are located in "Model" and "Model/fp16" respectively.

- Modified models can be found in "Modified-Model" and "Modified-Model/fp16".

----

# Disclaimer

- The creation of SFW and NSFW images is at the discretion of the individual creator.

- This model is not a model created to publish NSFW content in public places, etc.

----

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.

The CreativeML OpenRAIL License specifies:

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content

2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license

3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)

(Full text of the license: https://huggingface.co/spaces/CompVis/stable-diffusion-license)