---
license: creativeml-openrail-m
---
<b>This model is available on <a href="https://www.mage.space/">Mage.Space</a> (main sponsor)</b><br>

<b>Please read this!</b><br>
For version 2.0 it is recommended to use with VAE (to improve generation quality and get rid of blue artifacts): https://huggingface.co/stabilityai/sd-vae-ft-mse-original<br>

<hr/>

<b>I use this template to get good generation results:

Prompt:</b>
RAW photo, *subject*, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3

<b>Example:</b> RAW photo, a close up portrait photo of 26 y.o woman in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3


<b>Negative Prompt:</b>
(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck<br>

<b>OR</b><br>

(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation

<b>Euler A or DPM++ 2M Karras with 25 steps<br>
CFG Scale 3,5 - 7<br>
Hires. fix with Latent upscaler<br>
0 Hires steps and Denoising strength 0.25-0.45<br>
Upscale by 1.1-2.0</b>