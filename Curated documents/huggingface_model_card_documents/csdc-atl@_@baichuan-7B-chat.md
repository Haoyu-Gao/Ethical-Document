---
language:
- zh
license: apache-2.0
---

# baichuan-7B-chat

## 介绍
baichuan-7B-chat是在私有对话指令数据上微调的多轮对话模型。

- 基座模型是[baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B)，是由百川智能开发的一个开源可商用的大规模预训练语言模型。
- 训练数据为私有多轮对话指令数据。

## 代码调用

可以通过如下代码调用 baichuan-7B-chat 模型来生成对话：

```ipython
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("csdc-atl/baichuan-7B-chat", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("csdc-atl/baichuan-7B-chat", trust_remote_code=True).half().cuda()
>>> response, history = model.chat(tokenizer, "“面朝大海，春暖花开”的出处是？", history=[])
>>> print(response)
“面朝大海，春暖花开”是一句著名的诗歌，出自中国现代诗人海子之手。这首诗表达了对大海的热爱和对春天的向往，同时也反映了诗人对生命和自然的感悟和思考。这首诗被广泛传诵和引用，成为了中国现代诗歌中的经典之作。
>>> response, history = model.chat(tokenizer, "能不能把这一首诗完整背诵一下", history=history)
>>> print(response)
当然可以，以下是“面朝大海，春暖花开”的原文：
面朝大海，春暖花开
从明天起，做一个幸福的人
喂马，劈柴，周游世界
从明天起，关心粮食和蔬菜
我有一所房子，面朝大海，春暖花开
从明天起，和每一个亲人通信
告诉他们我的幸福
那幸福的闪电告诉我的
我将告诉每一个人
给每一条河每一座山取一个温暖的名字
陌生人，我也为你祝福
愿你有一个灿烂的前程
愿你有情人终成眷属
愿你在尘世获得幸福
我只愿面朝大海，春暖花开
```

## 局限性

本项目中的模型存在局限性，包括但不限于：
- 在面对事实性知识任务时，可能会生成不正确的信息或者产生不稳定的输出（有时可以返回正确答案，有时不能）。

## 协议

本仓库的代码依照 [Apache-2.0](LICENSE) 协议开源。

baichuan-7B-chat模型支持商用。但按照baichuan-7B的要求，如果将baichuan-7B衍生品用作商业用途，需要联系[baichuan-7B的许可方](https://github.com/baichuan-inc/baichuan-7B#%E5%8D%8F%E8%AE%AE)。