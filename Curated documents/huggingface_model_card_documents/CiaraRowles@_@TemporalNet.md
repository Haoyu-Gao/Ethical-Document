---
license: openrail
tags:
- controlnet
- stable-diffusion
- diffusers
base_model: runwayml/stable-diffusion-v1-5
---
Introducing the Beta Version of TemporalNet

TemporalNet is a ControlNet model designed to enhance the temporal consistency of generated outputs, as demonstrated in this example: https://twitter.com/CiaraRowles1/status/1637486561917906944. While it does not eliminate all flickering, it significantly reduces it, particularly at higher denoise levels. For optimal results, it is recommended to use TemporalNet in combination with other methods.

Instructions for Use:

1) Add the model "diff_control_sd15_temporalnet_fp16.safetensors" to your models folder in the ControlNet extension in Automatic1111's Web UI.

2) Create a folder that contains:

- A subfolder named "Input_Images" with the input frames
- A PNG file called "init.png" that is pre-stylized in your desired style
- The "temporalvideo.py" script

3) Customize the "temporalvideo.py" script according to your preferences, such as the image resolution, prompt, and control net settings.

4) Launch Automatic1111's Web UI with the --api setting enabled.

5) Execute the Python script.

*Please note that the "init.png" image will not significantly influence the style of the output video. Its primary purpose is to prevent a drastic change in aesthetics during the first few frames.*

Also, I highly recommend you use this in conjunction with the hed model, the settings are already in the script.

ToDo:

Write an Extension for the web ui.

Write a feature that automatically generates an "init.png" image if none is provided.

 ̶C̶h̶a̶n̶g̶e̶ ̶t̶h̶e̶ ̶e̶x̶t̶e̶n̶s̶i̶o̶n̶ ̶t̶o̶ ̶.̶s̶a̶f̶e̶t̶e̶n̶s̶o̶r̶s̶ ̶a̶n̶d̶ ̶i̶n̶v̶e̶s̶t̶i̶g̶a̶t̶e̶ ̶c̶o̶m̶p̶r̶e̶s̶s̶i̶o̶n̶.̶
