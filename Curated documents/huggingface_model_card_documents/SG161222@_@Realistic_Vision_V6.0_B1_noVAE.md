---
license: creativeml-openrail-m
---
<b>This model is available on <a href="https://www.mage.space/">Mage.Space</a> (main sponsor)</b><br>

<b>Please read this!</b><br>
This is not yet the full version of the model (read the <b>"Model Description"</b> section).<br>
For version 6.0 it is recommended to use with VAE (to improve generation quality and get rid of artifacts): https://huggingface.co/stabilityai/sd-vae-ft-mse-original<br>

<b>Model Description</b><br>
Realistic Vision V6.0 "New Vision" is a global update for the Realistic Vision model, which will be released gradually in several beta versions until the full release. The model is aimed at realism and photorealism.<br>
CivitAI Page: https://civitai.com/models/4201/realistic-vision-v60-b1?modelVersionId=245598

<b>Resolutions (use lower resolution if you get a lot of mutations and stuff like that)</b><br>
- Face Portrait: 896x896<br>
- Portrait: 896x896, 768x1024<br>
- Half Body: 768x1024, 640x1152<br>
- Full Body: 896x896, 768x1024, 640x1152, 1024x768, 1152x640<br>

<b>Improvements</b>
- increased generation resolution to such resolutions as: 896x896, 768x1024, 640x1152, 1024x768, 1152x640. (note. in some cases there may still be mutations, duplications, etc -> will be fixed in future versions).<br>
- improved sfw and nsfw for female and female anatomy (note. not all poses work correctly in such large resolutions -> will be fixed in future versions).<br>

<b>Recommended Workflow</b><br>
Images can be generated with or without Hires.Fix, but it will help improve the generation quality significantly. In some cases it is strictly recommended to use Hires.Fix, namely when generating full body and half body images (note: you can also use Restore Faces or ADetailer).<br>

<b>Recommended Generation Parameters</b><br>
Sampler: DPM++ SDE Karras (25+ steps) / DPM++ 2M SDE (50+ steps)<br>
Negative Prompt: (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck<br>

<b>Recommended Hires.Fix Parameters</b><br>
Sampler: DPM++ SDE Karras or DPM++ 2M SDE<br>
Denoising steps: 10+ (DPM++ SDE Karras) / 20+ (DPM++ 2M SDE (notice. the lower the value of hires steps at a given sampler, the stronger the skin texture and the higher the chance of getting artifacts))<br>
Denoising strength: 0.1-0.3<br>
Upscaler: 4x-UltraSharp / 4x_NMKD-Superscale-SP_178000_G or another<br>
Upscale by: 1.1-2.0+<br>
