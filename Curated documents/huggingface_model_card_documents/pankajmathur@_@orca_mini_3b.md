---
language:
- en
license: cc-by-nc-sa-4.0
library_name: transformers
datasets:
- psmathur/alpaca_orca
- psmathur/dolly-v2_orca
- psmathur/WizardLM_Orca
pipeline_tag: text-generation
---
# orca_mini_3b

Use orca-mini-3b on Free Google Colab with T4 GPU :)

<a target="_blank" href="https://colab.research.google.com/#fileId=https://huggingface.co/psmathur/orca_mini_3b/blob/main/orca_mini_3b_T4_GPU.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

An [OpenLLaMa-3B model](https://github.com/openlm-research/open_llama) model trained on explain tuned datasets, created using Instructions and Input from WizardLM, Alpaca & Dolly-V2 datasets and applying Orca Research Paper dataset construction approaches.


# Dataset

We build explain tuned [WizardLM dataset ~70K](https://github.com/nlpxucan/WizardLM), [Alpaca dataset ~52K](https://crfm.stanford.edu/2023/03/13/alpaca.html)  & [Dolly-V2 dataset ~15K](https://github.com/databrickslabs/dolly) created using approaches from [Orca Research Paper](https://arxiv.org/abs/2306.02707).

We leverage all of the 15 system instructions provided in Orca Research Paper. to generate custom datasets, in contrast to vanilla instruction tuning approaches used by original datasets.

This helps student model aka this model to learn ***thought*** process from teacher model, which is ChatGPT (gpt-3.5-turbo-0301 version).

Please see below example usage how the **System** prompt is added before each **instruction**.


# Training

The training configurations are provided in the table below.

The training takes on 8x A100(80G) GPUs and lasts for around 4 Hours for cost of $48 using [Lambda Labs](https://lambdalabs.com)

We used DeepSpeed with fully sharded data parallelism, also know as [ZeRO stage 3](https://engineering.fb.com/2021/07/15/open-source/fsdp/) by writing our own fine tunning scripts plus leveraging some of the model training code provided by amazing [OpenAlpaca repo](https://github.com/yxuansu/OpenAlpaca)

Here are some of params used during training:

|||
|:-------------:|:-------------:|
|*batch_size*|64|
|*train_micro_batch_size_per_gpu*|4|
|*gradient_accumulation_steps*|2|
|*Learning rate*|2e-5|
|*Max length*|1024|
|*Epochs*|3|
|*Optimizer*|AdamW|



# Example Usage

Below shows an example on how to use this model

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Hugging Face model_path
model_path = 'psmathur/orca_mini_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)


#generate text function
def generate_text(system, instruction, input=None):
    
    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'],
            temperature=instance['temperature'],
            top_k=instance['top_k']
        )    
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f'[!] Response: {string}'

# Sample Test Instruction Used by Youtuber Sam Witteveen https://www.youtube.com/@samwitteveenai
system = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'
instruction = 'Write a letter to Sam Altman, CEO of OpenAI, requesting him to convert GPT4 a private model by OpenAI to an open source project'
print(generate_text(system, instruction))

```

```

[!] Response:
Dear Sam Altman,

I am writing to request that you convert the GPT4 private model developed by OpenAI to an open source project. As a user of OpenAI, I have been waiting for the day when I can use the advanced natural language processing capabilities of GPT4 in a more open and accessible way.

While OpenAI has made significant progress in developing AI applications, it has primarily focused on building private models that are not accessible to the general public. However, with the recent release of GPT-3, there is a growing demand for more open and accessible AI tools.

Converting GPT4 to an open source project would allow for greater transparency, collaboration, and innovation. It would also help to build trust in the technology and ensure that it is used ethically and responsibly.

I urge you to consider converting GPT4 to an open source project. This would be a significant contribution to the AI community and would help to create a more open and accessible future.

Thank you for your consideration.

Sincerely,

[Your Name]

```


**P.S. I am #opentowork and #collaboration, if you can help, please reach out to me at www.linkedin.com/in/pankajam**


Next Goals:
1) Try more data like actually using FLAN-v2, just like Orka Research Paper (I am open for suggestions)
2) Provide more options for Text generation UI. (may be https://github.com/oobabooga/text-generation-webui)
3) Provide 4bit GGML/GPTQ quantized model (may be [TheBloke](https://huggingface.co/TheBloke) can help here)


Limitations & Biases:

This model can produce factually incorrect output, and should not be relied on to produce factually accurate information.
This model was trained on various public datasets. While great efforts have been taken to clean the pretraining data, it is possible that this model could generate lewd, biased or otherwise offensive outputs.

Disclaimer:

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model.
Please cosult an attorney before using this model for commercial purposes.


Citiation:

If you found wizardlm_alpaca_dolly_orca_open_llama_3b useful in your research or applications, please kindly cite using the following BibTeX:

```
@misc{orca_mini_3b,
  author = {Pankaj Mathur},
  title = {wizardlm_alpaca_dolly_orca_open_llama_3b: An explain tuned OpenLLaMA-3b model on custom wizardlm, alpaca, & dolly datasets},
  year = {2023},
  publisher = {GitHub, HuggingFace},
  journal = {GitHub repository, HuggingFace repository},
  howpublished = {\url{https://github.com/pankajarm/wizardlm_alpaca_dolly_orca_open_llama_3b}, \url{https://https://huggingface.co/psmathur/wizardlm_alpaca_dolly_orca_open_llama_3b}},
}
```
```
@misc{mukherjee2023orca,
      title={Orca: Progressive Learning from Complex Explanation Traces of GPT-4}, 
      author={Subhabrata Mukherjee and Arindam Mitra and Ganesh Jawahar and Sahaj Agarwal and Hamid Palangi and Ahmed Awadallah},
      year={2023},
      eprint={2306.02707},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@software{openlm2023openllama,
  author = {Xinyang Geng and Hao Liu},
  title = {OpenLLaMA: An Open Reproduction of LLaMA},
  month = May,
  year = 2023,
  url = {https://github.com/openlm-research/open_llama}
}
```
```
@misc{openalpaca,
  author = {Yixuan Su and Tian Lan and Deng Cai},
  title = {OpenAlpaca: A Fully Open-Source Instruction-Following Model Based On OpenLLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yxuansu/OpenAlpaca}},
}
```
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_psmathur__orca_mini_3b)

| Metric                | Value                     |
|-----------------------|---------------------------|
| Avg.                  | 35.5   |
| ARC (25-shot)         | 41.55          |
| HellaSwag (10-shot)   | 61.52    |
| MMLU (5-shot)         | 26.79         |
| TruthfulQA (0-shot)   | 42.42   |
| Winogrande (5-shot)   | 61.8   |
| GSM8K (5-shot)        | 0.08        |
| DROP (3-shot)         | 14.33         |
