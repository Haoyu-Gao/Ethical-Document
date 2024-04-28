---
license: creativeml-openrail-m
tags:
- stablediffusionapi.com
- stable-diffusion-api
- text-to-image
- ultra-realistic
pinned: true
---

# Realistic Stock Photo API Inference

![generated from stablediffusionapi.com](https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/14450003171697899771.png)
## Get API Key

Get API key from [Stable Diffusion API](http://stablediffusionapi.com/), No Payment needed. 

Replace Key in below code, change **model_id**  to "realistic-stock-photo"

Coding in PHP/Node/Java etc? Have a look at docs for more code examples: [View docs](https://stablediffusionapi.com/docs)

Try model for free: [Generate Images](https://stablediffusionapi.com/models/realistic-stock-photo)

Model link: [View model](https://stablediffusionapi.com/models/realistic-stock-photo)

Credits: [View credits](https://civitai.com/?query=Realistic%20Stock%20Photo)

View all models: [View Models](https://stablediffusionapi.com/models)

    import requests  
    import json  
      
    url =  "https://stablediffusionapi.com/api/v4/dreambooth"  
      
    payload = json.dumps({  
    "key":  "your_api_key",  
    "model_id":  "realistic-stock-photo",  
    "prompt":  "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner)), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K",  
    "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
    "width":  "512",  
    "height":  "512",  
    "samples":  "1",  
    "num_inference_steps":  "30",  
    "safety_checker":  "no",  
    "enhance_prompt":  "yes",  
    "seed":  None,  
    "guidance_scale":  7.5,  
    "multi_lingual":  "no",  
    "panorama":  "no",  
    "self_attention":  "no",  
    "upscale":  "no",  
    "embeddings":  "embeddings_model_id",  
    "lora":  "lora_model_id",  
    "webhook":  None,  
    "track_id":  None  
    })  
      
    headers =  {  
    'Content-Type':  'application/json'  
    }  
      
    response = requests.request("POST", url, headers=headers, data=payload)  
      
    print(response.text)

> Use this coupon code to get 25% off **DMGG0RBN** 