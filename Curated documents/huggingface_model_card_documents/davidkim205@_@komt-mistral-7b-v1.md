---
language:
- en
- ko
tags:
- finetuned
pipeline_tag: text-generation
---
# komt : korean multi task instruction tuning model
![multi task instruction tuning.jpg](https://github.com/davidkim205/komt/assets/16680469/c7f6ade7-247e-4b62-a94f-47e19abea68e)

Recently, due to the success of ChatGPT, numerous large language models have emerged in an attempt to catch up with ChatGPT's capabilities. 
However, when it comes to Korean language performance, it has been observed that many models still struggle to provide accurate answers or generate Korean text effectively. 
This study addresses these challenges by introducing a multi-task instruction technique that leverages supervised datasets from various tasks to create training data for Large Language Models (LLMs).

## Model Details

* **Model Developers** :  davidkim(changyeon kim)
* **Repository** : https://github.com/davidkim205/komt
* **Model Architecture** : The komt-mistral-7b-v1  is is a fine-tuned version of the Mistral-7B-Instruct-v0.1.

## Dataset
korean multi-task instruction dataset

## Hardware and Software
- nvidia driver : 535.54.03
- CUDA Version: 12.2

## Training
Refer https://github.com/davidkim205/komt


## Prompt template: Mistral
```
<s>[INST] {prompt} [/INST]</s>
```

## Usage


``` 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer, GenerationConfig

model_name='davidkim205/komt-mistral-7b-v1'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

def gen(x):
    generation_config = GenerationConfig(
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
    )
    q = f"[INST]{x} [/INST]"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"\n\n### Response: "
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

print(gen('제주도를 1박2일로 혼자 여행하려고 하는데 여행 코스를 만들어줘'))
```
output
``` 
<s> [INST]제주도를 1박2일로 혼자 여행하려고 하는데 여행 코스를 만들어줘 [/INST]1박 2일 여행 코스
제주 여행 코스는 다음과 같습니다:

첫째 날:
* 아침: 제주시 신라로 이동하여 한라산 국립공원에 도착합니다. 오른쪽 발자국 광장에서 눈에 띄는 맥주숍을 찾아 한 병의 맥주를 마시고 계단으로 올라가 한라산 정상에 도착합니다.
* 점심: 정상에서 계단을 내려와 중도에 위치한 골목에서 음식을 즐깁니다. 젓갈, 곰탕, 전복 등 맛있는 제주 특산물을 맛보며 힐링을 즐깁니다.
* 오후: 중도에서 남하하여 남서쪽 제주로 이동합니다. 남서쪽 제주는 섬의 자연 경관과 백사장, 아름다운 바다 등 아름다운 풍광이 있는 곳입니다. 숙소에 숙박합니다.

둘째 날:
* 아침: 남서쪽 제주에서 북서쪽 제주로 이동합니다. 이 지역은 푸른 잔디 해변과 투명한 바다가 펼쳐져 있는 아름다운 풍광입니다. 숙소에서 아침을 먹고 바다로 향하여 해변에서 휴식을 취합니다.
* 점심: 북서쪽 제주의 앞바다에서 시원한 바다를 보며 한 끼의 해산물을 맛보고 계절에 따라 해산물 요리를 즐깁니다.
* 오후: 남서쪽 제주에서 이동하여 서쪽 제주로 이동합니다. 서쪽 제주는 절벽과 암석 해변, 수수께끼의 돌 정원 등 독특한 경관이 있는 곳입니다. 이곳에서는 서쪽 제주의 대표적인 명소인 청량사를 방문하고 서쪽 제주의 아름다운 풍광을 감상합니다.
* 저녁: 서쪽 제주에서 제주 시로 이동하여 한 끼의 제주 특산물을 맛보고 도착한 제주 도심에서 저녁을 즐깁니다.
* 야간: 제주 시의 도심에서 야간 활동을 즐기며 1박 2일의 여행을 마무리합니다.

이렇게 제주를 1박 2일로 혼자 여행하면 제주의 아름다운 풍광, 푸른 잔디 해변, 투명한 바다 등을 경험할 수 있습니다. 
```
## Evaluation
For objective model evaluation, we initially used EleutherAI's lm-evaluation-harness but obtained unsatisfactory results. Consequently, we conducted evaluations using ChatGPT, a widely used model, as described in [Self-Alignment with Instruction Backtranslation](https://arxiv.org/pdf/2308.06502.pdf) and [Three Ways of Using Large Language Models to Evaluate Chat](https://arxiv.org/pdf/2308.06259.pdf) .


| model                                   | score   | average(0~5) | percentage |
| --------------------------------------- |---------| ------------ | ---------- |
| gpt-3.5-turbo(close)                    | 147     | 3.97         | 79.45%     |
| naver Cue(close)                        | 140     | 3.78         | 75.67%     |
| clova X(close)                          | 136     | 3.67         | 73.51%     |
| WizardLM-13B-V1.2(open)                 | 96      | 2.59         | 51.89%     |
| Llama-2-7b-chat-hf(open)                | 67      | 1.81         | 36.21%     |
| Llama-2-13b-chat-hf(open)               | 73      | 1.91         | 38.37%     |
| nlpai-lab/kullm-polyglot-12.8b-v2(open) | 70      | 1.89         | 37.83%     |
| kfkas/Llama-2-ko-7b-Chat(open)          | 96      | 2.59         | 51.89%     |
| beomi/KoAlpaca-Polyglot-12.8B(open)     | 100     | 2.70         | 54.05%     |
| **komt-llama2-7b-v1 (open)(ours)**      | **117** | **3.16**     | **63.24%** |
| **komt-llama2-13b-v1  (open)(ours)**    | **129** | **3.48**     | **69.72%** |
| **komt-llama-30b-v1  (open)(ours)**    | **129** | **3.16**     | **63.24%** |
| **komt-mistral-7b-v1  (open)(ours)**    | **131** | **3.54**     | **70.81%** |

