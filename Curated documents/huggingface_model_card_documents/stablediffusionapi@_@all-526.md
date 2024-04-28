---
license: creativeml-openrail-m
tags:
- stablediffusionapi.com
- stable-diffusion-api
- text-to-image
- ultra-realistic
pinned: true
---

# All 526 API Inference

![generated from stablediffusionapi.com](https://pub-8b49af329fae499aa563997f5d4068a4.r2.dev/generations/9290108341682539305.png)
## Get API Key

Get API key from [Stable Diffusion API](http://stablediffusionapi.com/), No Payment needed. 

Replace Key in below code, change **model_id**  to "all-526"

Coding in PHP/Node/Java etc? Have a look at docs for more code examples: [View docs](https://stablediffusionapi.com/docs)

Model link: [View model](https://stablediffusionapi.com/models/all-526)

Credits: [View credits](https://civitai.com/?query=All%20526)

View all models: [View Models](https://stablediffusionapi.com/models)

    import requests  
    import json  
      
    url =  "https://stablediffusionapi.com/api/v3/dreambooth"  
      
    payload = json.dumps({  
    "key":  "",  
    "model_id":  "all-526",  
    "prompt":  "actual 8K portrait photo of gareth person, portrait, happy colors, bright eyes, clear eyes, warm smile, smooth soft skin, big dreamy eyes, beautiful intricate colored hair, symmetrical, anime wide eyes, soft lighting, detailed face, by makoto shinkai, stanley artgerm lau, wlop, rossdraws, concept art, digital painting, looking into camera",  
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