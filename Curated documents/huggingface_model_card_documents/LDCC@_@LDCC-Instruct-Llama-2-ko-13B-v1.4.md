---
language:
- ko
license: cc-by-nc-4.0
---

# Model Card for LDCC-Instruct-Llama-2-ko-13B-v1.4 

LDCC-Instruct-Llama-2-ko-13B-v1.4 is a continuation in a series of language models designed to serve as efficient assistants. This fifth iteration is an enhanced version of its predecessor, [LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.0](https://huggingface.co/LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.0). We applied [NEFTune](https://arxiv.org/abs/2310.05914) noise embeddings to fine-tuning. This has been proven to improve model performances for instrcution fine-tuning. Additionally, it underwent fine-tuning on a combination of publicly available and synthetic datasets through the use of [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290). Interestingly, we observed an uplift in performance on the MT Bench when the intrinsic alignment of these datasets was eliminated, resulting in a more effective assistant model.

## Developed by : Wonchul Kim ([Lotte Data Communication](https://www.ldcc.co.kr) AI Technical Team)

## Hardware and Software

* **Hardware**: We utilized an A100x8 * 1 for training our model
* **Training Factors**: We fine-tuned this model using a combination of the [DeepSpeed library](https://github.com/microsoft/DeepSpeed) and the [HuggingFace TRL Trainer](https://huggingface.co/docs/trl/trainer) / [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index)

## Base Model : [beomi/llama-2-koen-13b](https://huggingface.co/beomi/llama-2-koen-13b)

### Training Data

The LDCC-Instruct-Llama-2-ko-13B model was trained with publicly accessible Korean/English data sources. For its fine-tuning, we utilized other public data and underwent some processing and refinement.

We did not incorporate any client data owned by Lotte Data Communication.

## Prompt Template
```
### Prompt:
{instruction}

### Answer:
{output}
```

### License
[LICENSE.txt](LICENSE.txt)