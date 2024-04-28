---
language:
- en
license: openrail++
tags:
- stable-diffusion
- controlnet
- qrcode
---
# Controlnet QR Code Monster v2 For SD-1.5

![QR code in shape of a blue monster, reading "https://qrcode.monster"](images/monster.png)

##  Model Description

This model is made to generate creative QR codes that still scan.
Keep in mind that not all generated codes might be readable, but you can try different parameters and prompts to get the desired results.

**NEW VERSION**

Introducing the upgraded version of our model - Controlnet QR code Monster v2.
V2 is a huge upgrade over v1, for scannability AND creativity.

QR codes can now seamlessly blend the image by using a gray-colored background (#808080).

As with the former version, the readability of some generated codes may vary, however playing around with parameters and prompts could yield better results.

You can find in in the `v2/` subfolder.

## How to Use

- **Condition**: QR codes are passed as condition images with a module size of 16px. Use a higher error correction level to make it easier to read (sometimes a lower level can be easier to read if smaller in size). Use a gray background for the rest of the image to make the code integrate better.

- **Prompts**: Use a prompt to guide the QR code generation. The output will highly depend on the given prompt. Some seem to be really easily accepted by the qr code process, some will require careful tweaking to get good results.

- **Controlnet guidance scale**: Set the controlnet guidance scale value:
   - High values: The generated QR code will be more readable.
   - Low values: The generated QR code will be more creative.

### Tips

- For an optimally readable output, try generating multiple QR codes with similar parameters, then choose the best ones.

- Use the Image-to-Image feature to improve the readability of a generated QR code:
  - Decrease the denoising strength to retain more of the original image.
  - Increase the controlnet guidance scale value for better readability.
  A typical workflow for "saving" a code would be :
  Max out the guidance scale and minimize the denoising strength, then bump the strength until the code scans.

## Example Outputs

Here are some examples of creative, yet scannable QR codes produced by our model:

![City ruins with a building facade in shape of a QR code, reading "https://qrcode.monster"](images/architecture.png)
![QR code in shape of a tree, reading "https://qrcode.monster"](images/tree.png)
![A gothic sculpture in shape of a QR code, reading "https://qrcode.monster"](images/skulls.png)

Feel free to experiment with prompts, parameters, and the Image-to-Image feature to achieve the desired QR code output. Good luck and have fun!