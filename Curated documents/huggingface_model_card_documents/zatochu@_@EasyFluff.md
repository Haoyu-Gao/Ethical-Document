---
{}
---
V10-FunnerEdition?

- Tweaked UNET with supermerger adjust to dial back noise/detail that can resolve eye sclera bleed in some cases.
- Adjusted contrast and color temperature. (Less orange/brown by default)
- CLIP should theoretically respond more to natural language. (Don't conflate this with tags not working or having to use natural language. Also it is not magic, so don't expect extremely nuanced prompts to work better.)
- FunEdition and FunEditionAlt are earlier versions before adjusting the UNET further to fix color temperature and color bleed. CLIP on these versions may be less predictable as well.

HOW TO RUN THIS MODEL

- This is a terminal-snr-v-prediction model and you will need an accompanying configuration file to load the checkpoint in v-prediction mode. Relevant configuration files are available in this repository. Place them in the same folder as the checkpoint. ComfyUI users will need to place this configuration file in models/configs and use the Load Checkpoint (With Config) node.

- You will also need https://github.com/Seshelle/CFG_Rescale_webui. This extension can be installed from the Extensions tab by copying this repository link into the Install from URL section. A CFG Rescale value of 0.7 is recommended by the creator of the extension themself. The CFG Rescale slider will be below your generation parameters and above the scripts section when installed. If you do not do this and run inference without CFG Rescale, these will be the types of results you can expect per this research paper. https://arxiv.org/pdf/2305.08891.pdf

  <img src="https://huggingface.co/zatochu/EasyFluff/resolve/main/aaef6b3f-8cde-4a34-a4ae-6b7a066a3766.png">

- If you are on ComfyUI, you will need the sampler_rescalecfg.py node from https://github.com/comfyanonymous/ComfyUI_experiments. Same value recommendation applies.