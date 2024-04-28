---
license: apache-2.0
datasets:
- tatsu-lab/alpaca
---

## 🍮 🦙 Flan-Alpaca: Instruction Tuning from Humans and Machines

📣 Introducing **Red-Eval** to evaluate the safety of the LLMs using several jailbreaking prompts. With **Red-Eval** one could jailbreak/red-team GPT-4 with a 65.1% attack success rate and ChatGPT could be jailbroken 73% of the time as measured on DangerousQA and HarmfulQA benchmarks. More details are here: [Code](https://github.com/declare-lab/red-instruct) and [Paper](https://arxiv.org/abs/2308.09662).

📣 We developed Flacuna by fine-tuning Vicuna-13B on the Flan collection. Flacuna is better than Vicuna at problem-solving. Access the model here https://huggingface.co/declare-lab/flacuna-13b-v1.0.

📣 Curious to know the performance of 🍮 🦙 **Flan-Alpaca** on large-scale LLM evaluation benchmark, **InstructEval**? Read our paper [https://arxiv.org/pdf/2306.04757.pdf](https://arxiv.org/pdf/2306.04757.pdf). We evaluated more than 10 open-source instruction-tuned LLMs belonging to various LLM families including Pythia, LLaMA, T5, UL2, OPT, and Mosaic. Codes and datasets: [https://github.com/declare-lab/instruct-eval](https://github.com/declare-lab/instruct-eval)


📣 **FLAN-T5** is also useful in text-to-audio generation. Find our work at [https://github.com/declare-lab/tango](https://github.com/declare-lab/tango) if you are interested.


Our [repository](https://github.com/declare-lab/flan-alpaca) contains code for extending the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
synthetic instruction tuning to existing instruction-tuned models such as [Flan-T5](https://arxiv.org/abs/2210.11416).
We have a [live interactive demo](https://huggingface.co/spaces/joaogante/transformers_streaming) thanks to [Joao Gante](https://huggingface.co/joaogante)!
We are also benchmarking many instruction-tuned models at [declare-lab/flan-eval](https://github.com/declare-lab/flan-eval).
Our pretrained models are fully available on HuggingFace 🤗 :

| Model                                                                            | Parameters | Instruction Data                                                                                                                                   | Training GPUs   |
|----------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| [Flan-Alpaca-Base](https://huggingface.co/declare-lab/flan-alpaca-base)          | 220M       | [Flan](https://github.com/google-research/FLAN), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                                            | 1x A6000        |
| [Flan-Alpaca-Large](https://huggingface.co/declare-lab/flan-alpaca-large)        | 770M       | [Flan](https://github.com/google-research/FLAN), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                                            | 1x A6000        |
| [Flan-Alpaca-XL](https://huggingface.co/declare-lab/flan-alpaca-xl)              | 3B         | [Flan](https://github.com/google-research/FLAN), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                                            | 1x A6000        |
| [Flan-Alpaca-XXL](https://huggingface.co/declare-lab/flan-alpaca-xxl)            | 11B        | [Flan](https://github.com/google-research/FLAN), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                                            | 4x A6000 (FSDP) |
| [Flan-GPT4All-XL](https://huggingface.co/declare-lab/flan-gpt4all-xl)            | 3B         | [Flan](https://github.com/google-research/FLAN), [GPT4All](https://github.com/nomic-ai/gpt4all)                                                    | 1x A6000        |
| [Flan-ShareGPT-XL](https://huggingface.co/declare-lab/flan-sharegpt-xl)          | 3B         | [Flan](https://github.com/google-research/FLAN), [ShareGPT](https://github.com/domeccleston/sharegpt)/[Vicuna](https://github.com/lm-sys/FastChat) | 1x A6000        |
| [Flan-Alpaca-GPT4-XL*](https://huggingface.co/declare-lab/flan-alpaca-gpt4-xl)   | 3B         | [Flan](https://github.com/google-research/FLAN), [GPT4-Alpaca](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)                         | 1x A6000        |

*recommended for better performance

### Why?

[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) represents an exciting new direction
to approximate the performance of large language models (LLMs) like ChatGPT cheaply and easily.
Concretely, they leverage an LLM such as GPT-3 to generate instructions as synthetic training data.
The synthetic data which covers more than 50k tasks can then be used to finetune a smaller model.
However, the original implementation is less accessible due to licensing constraints of the
underlying [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) model.
Furthermore, users have noted [potential noise](https://github.com/tloen/alpaca-lora/issues/65) in the synthetic
dataset. Hence, it may be better to explore a fully accessible model that is already trained on high-quality (but
less diverse) instructions such as [Flan-T5](https://arxiv.org/abs/2210.11416).

### Usage

```
from transformers import pipeline

prompt = "Write an email about an alpaca that likes flan"
model = pipeline(model="declare-lab/flan-alpaca-gpt4-xl")
model(prompt, max_length=128, do_sample=True)

# Dear AlpacaFriend,
# My name is Alpaca and I'm 10 years old.
# I'm excited to announce that I'm a big fan of flan!
# We like to eat it as a snack and I believe that it can help with our overall growth.
# I'd love to hear your feedback on this idea. 
# Have a great day! 
# Best, AL Paca
```