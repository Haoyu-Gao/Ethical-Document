---
library_name: diffusers
tags:
- animatediff
pipeline_tag: text-to-video
---

# Motion LoRAs

Motion LoRAs allow adding specific types of motion to your animations. 

![animatediff-zoom-out-lora.gif](https://cdn-uploads.huggingface.co/production/uploads/6126e46848005fa9ca5c578c/13B2HSVUuZ1t9UseffdHp.gif)



Currently the following types of motion are available for models using the `guoyww/animatediff-motion-adapter-v1-5-2` checkpoint.

- Zoom In/Out
- Pan Left/Right
- Tilt Up/Down
- Rolling Clockwise/Anticlockwise

Please refer to the [AnimateDiff documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/animatediff) for information on how to use these Motion LoRAs.