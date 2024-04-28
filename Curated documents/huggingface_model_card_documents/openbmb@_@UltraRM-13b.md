---
license: mit
datasets:
- openbmb/UltraFeedback
---

# News

- [2023/09/26]: UltraRM unleashes the power of [UltraLM-13B-v2.0](https://huggingface.co/openbmb/UltraLM-13b-v2.0) and [UltraLM-13B](https://huggingface.co/openbmb/UltraLM-13b)! A simple best-of-16 sampling achieves **92.30%** (UltraLM2, 🥇 in 13B results) and **91.54%** (UltraLM, 🥇 in LLaMA-1 results) win rates against text-davinci-003 on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) benchmark!
- [2023/09/26]: We release the [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset, along with UltraFeedback-powered reward model [UltraRM](https://huggingface.co/datasets/openbmb/UltraFeedback) and critique model [UltraCM](https://huggingface.co/datasets/openbmb/UltraCM-13b)! Both built **new SOTAs** over open-source models!  

# Links

- 🤗 [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- 🤗 [UltraRM](https://huggingface.co/datasets/openbmb/UltraRM-13b)
- 🤗 [UltraCM](https://huggingface.co/datasets/openbmb/UltraCM-13b)

# UltraRM

We train and release a reward model UltraRM based on UltraFeedback to further facilitate alignment research. UltraRM is initialized by LLaMA2-13B.

Specifically, we train two versions of reward models, where UltraRM-UF is merely fine-tuned on UltraFeedback and UltraRM is fine-tuned on a mixture of UltraFeedback and an equal-size sample from three open-source datasets including [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), [Standford SHP](https://huggingface.co/datasets/stanfordnlp/SHP), and [Summarization](https://huggingface.co/datasets/openai/summarize_from_feedback).

## Reward Modeling

On four public preference test sets, our UltraRM achieves SOTA over other open-source reward models. 

## Usage
```python
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards

ultrarm_template = """Human: {instruction}

Assistant: {completion}"""

dataset = [ # cases in webgpt
  {"chosen":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Sean Connery's character, fictional British Secret Service agent James Bond, in the movie Goldfinger, stated that he preferred his martini to be \"shaken, not stirred\". [1] Some believe that Bond ordered his martini shaken because of the vodka it contained, as vodka was, for the most part, refined from potatoes (cheaper brands) which made the vodka oily. To disperse the oil, Bond ordered his martinis shaken. [2]","rejected":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Fleming's fictional British Secret Service agent James Bond orders his martini cocktail shaken, not stirred [1]. Bond's preferences for his martini are carried over to the films, where his orders are seen in both the 1961 film Dr. No and the 2006 film Casino Royale [1, 2]. In both films, Bond's subordinates copy his order, telling the bartender to keep the fruit with their drinks [2]. However, in the 2006 film, Bond appears irritated when the bartender asks if he would like his drink shaken or stirred [2]."},
  {"chosen":"Human: Sural relates to which part of the body?\nAssistant: The sural region is the muscular swelling of the back of the leg below the knee, formed chiefly by the bellies of the gastrocnemius and soleus muscles [1,2].","rejected":"Human: Sural relates to which part of the body?\nAssistant: The Sural nerve runs down the side of the leg near the small saphenous vein, then passes forward below the lateral malleolus and continues on the outside of the foot as the lateral dorsal cutaneous nerve, which then communicates with the intermediate dorsal cutaneous nerve, which branches off to the side of the foot. [1]"}
]


tokenizer = LlamaTokenizer.from_pretrained("/data/UltraRM-13b")
model = LlamaRewardModel.from_pretrained("/data/UltraRM-13b")

for example in dataset:
    inputs = tokenizer(example["chosen"], return_tensors="pt")
    chosen_reward = model(**inputs).item()
    inputs = tokenizer(example["rejected"], return_tensors="pt")
    rejected_reward = model(**inputs).item()
    print(chosen_reward - rejected_reward)

# Output 1: 2.4158712085336447
# Output 2: 0.1896953582763672
```

## Citation
```
@misc{cui2023ultrafeedback,
      title={UltraFeedback: Boosting Language Models with High-quality Feedback}, 
      author={Ganqu Cui and Lifan Yuan and Ning Ding and Guanming Yao and Wei Zhu and Yuan Ni and Guotong Xie and Zhiyuan Liu and Maosong Sun},
      year={2023},
      eprint={2310.01377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```