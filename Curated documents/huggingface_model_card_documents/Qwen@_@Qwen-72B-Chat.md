---
language:
- zh
- en
tags:
- qwen
pipeline_tag: text-generation
inference: false
---

# Qwen-72B-Chat

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-72B-Chat-Demo/summary">Demo</a>
<br>
<a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp ｜  &nbsp&nbsp<a href="https://dashscope.aliyun.com">API</a> 
</p>
<br>

## 介绍（Introduction）

**通义千问-72B**（**Qwen-72B**）是阿里云研发的通义千问大模型系列的720亿参数规模的模型。Qwen-72B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-72B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-72B-Chat。本仓库为Qwen-72B-Chat的仓库。

通义千问-72B（Qwen-72B）主要有以下特点：

1. **大规模高质量训练语料**：使用超过3万亿tokens的数据进行预训练，包含高质量中、英、多语言、代码、数学等数据，涵盖通用及专业领域的训练语料。通过大量对比实验对预训练语料分布进行了优化。
2. **强大的性能**：Qwen-72B在多个中英文下游评测任务上（涵盖常识推理、代码、数学、翻译等），效果显著超越现有的开源模型。具体评测结果请详见下文。
3. **覆盖更全面的词表**：相比目前以中英词表为主的开源模型，Qwen-72B使用了约15万大小的词表。该词表对多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强和扩展。
4. **更长的上下文支持**：Qwen-72B支持32k的上下文长度。
5. **系统指令跟随**：Qwen-72B-Chat可以通过调整系统指令，实现**角色扮演**，**语言风格迁移**，**任务设定**，和**行为设定**等能力。

如果您想了解更多关于通义千问72B开源模型的细节，我们建议您参阅[GitHub代码库](https://github.com/QwenLM/Qwen)。

**Qwen-72B** is the 72B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-72B is a Transformer-based large language model, which is pretrained on a large volume of data, including web texts, books, codes, etc. Additionally, based on the pretrained Qwen-72B, we release Qwen-72B-Chat, a large-model-based AI assistant, which is trained with alignment techniques. This repository is the one for Qwen-72B-Chat.

The features of Qwen-72B include:

1. **Large-scale high-quality training corpora**: It is pretrained on over 3 trillion tokens, including Chinese, English, multilingual texts, code, and mathematics, covering general and professional fields. The distribution of the pre-training corpus has been optimized through a large number of ablation experiments.
2. **Competitive performance**: It significantly surpasses existing open-source models on multiple Chinese and English downstream evaluation tasks (including commonsense, reasoning, code, mathematics, etc.). See below for specific evaluation results.
3. **More comprehensive vocabulary coverage**: Compared with other open-source models based on Chinese and English vocabularies, Qwen-72B uses a vocabulary of over 150K tokens. This vocabulary is more friendly to multiple languages, enabling users to directly further enhance the capability for certain languages without expanding the vocabulary.
4. **Longer context support**: Qwen-72B supports 32k context length.
5. **System prompt**: Qwen-72B can realize roly playing, language style transfer, task setting, and behavior setting by using system prompt.

For more details about the open-source model of Qwen-72B, please refer to the [GitHub](https://github.com/QwenLM/Qwen) code repository.
<br>

## 要求（Requirements）

* python 3.8及以上版本
* pytorch 1.12及以上版本，推荐2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）
* **运行BF16或FP16模型需要多卡至少144GB显存（例如2xA100-80G或5xV100-32G）；运行Int4模型至少需要48GB显存（例如1xA100-80G或2xV100-32G）**
* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
* **To run Qwen-72B-Chat in bf16/fp16, at least 144GB GPU memory is required (e.g., 2xA100-80G or 5xV100-32G). To run it in int4, at least 48GB GPU memory is required (e.g., 1xA100-80G or 2xV100-32G)**
<br>

## 依赖项（Dependency）

### 使用HuggingFace进行推理

运行Qwen-72B-Chat，请确保满足上述要求，再执行以下pip命令安装依赖库

To run Qwen-72B-Chat, please make sure you meet the above requirements, and then execute the following pip commands to install the dependent libraries.

```bash
pip install "transformers>=4.32.0" accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
```

另外，推荐安装`flash-attention`库（**当前已支持flash attention 2**），以实现更高的效率和更低的显存占用。

In addition, it is recommended to install the `flash-attention` library (**we support flash attention 2 now.**) for higher efficiency and lower memory usage.

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# 如果你的flash-attn版本高于2.1.1，下方不需要安装。
# If the version of flash-attn is higher than 2.1.1, the following is not needed.
# pip install csrc/rotary
```

### 使用vLLM进行推理

使用vLLM进行推理可以支持更长的上下文长度并获得至少两倍的生成加速。你需要满足以下要求：

Using vLLM for inference can support longer context lengths and obtain at least twice the generation speedup. You need to meet the following requirements:

* pytorch >= 2.0
* cuda 11.8 or 12.1

如果你使用cuda12.1和pytorch2.1，可以直接使用以下命令安装vLLM。

If you use cuda 12.1 and pytorch 2.1, you can directly use the following command to install vLLM.

```bash
# pip install vllm  # This line is faster but it does not support quantization models.

# The below lines support int4 quantization (int8 will be supported soon). The installation are slower (~10 minutes).
git clone https://github.com/QwenLM/vllm-gptq
cd vllm-gptq
pip install -e .
```

否则请参考vLLM官方的[安装说明](https://docs.vllm.ai/en/latest/getting_started/installation.html)，或者我们[vLLM分支仓库（支持量化模型）](https://github.com/QwenLM/vllm-gptq)。

Otherwise, please refer to the official vLLM [Installation Instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html), or our [vLLM repo for GPTQ quantization](https://github.com/QwenLM/vllm-gptq).
<br>

## 快速使用（Quickstart）

### 使用HuggingFace Transformers进行推理（Inference with Huggingface Transformers）

下面我们展示了一个使用Qwen-72B-Chat模型，进行多轮对话交互的样例：

We show an example of multi-turn interaction with Qwen-72B-Chat in the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-72B-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-72B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-72B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-72B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-72B-Chat", device_map="auto", trust_remote_code=True).eval()
# NOTE: The above line would require at least 144GB memory in total

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-72B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 第三轮对话 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》

# Qwen-72B-Chat现在可以通过调整系统指令（System Prompt），实现角色扮演，语言风格迁移，任务设定，行为设定等能力。
# Qwen-72B-Chat can realize roly playing, language style transfer, task setting, and behavior setting by system prompt.
response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
print(response)
# 哎呀，你好哇！是怎么找到人家的呢？是不是被人家的魅力吸引过来的呀~(≧▽≦)/~

response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
print(response)
# Your colleague is a shining example of dedication and hard work. Their commitment to their job is truly commendable, and it shows in the quality of their work. 
# They are an asset to the team, and their efforts do not go unnoticed. Keep up the great work!
```



### 使用vLLM和类Transformers接口进行推理（Inference with vLLM and Transformers-like APIs）

在根据上方依赖性部分的说明安装vLLM后，可以下载[接口封装代码](https://qianwen-res.oss-cn-beijing.aliyuncs.com/vllm_wrapper.py)到当前文件夹，并执行以下命令进行多轮对话交互。（注意：该方法当前只支持``model.chat()``接口。）

After installing vLLM according to the dependency section above, you can download the [wrapper codes](https://qianwen-res.oss-cn-beijing.aliyuncs.com/vllm_wrapper.py) and execute the following commands for multiple rounds of dialogue interaction. (Note: It currently only supports the ``model.chat()`` method.)

```python
from vllm_wrapper import vLLMWrapper

model = vLLMWrapper('Qwen/Qwen-72B-Chat', tensor_parallel_size=2)
# model = vLLMWrapper('Qwen/Qwen-72B-Chat-Int4', tensor_parallel_size=1, dtype="float16")  # 运行int4模型。 run int4 model.

response, history = model.chat(query="你好", history=None)
print(response)
response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(query="给这个故事起一个标题", history=history)
print(response)
```

### 使用vLLM和类OpenAI接口进行推理（Inference with vLLM and OpenAI-like API）

请参考我们GitHub repo中[vLLM部署](https://github.com/QwenLM/Qwen#vllm)和[OpenAI接口使用](https://github.com/QwenLM/Qwen#openai-api)两个部分的介绍。

Please refer to the introduction of [vLLM deployment](https://github.com/QwenLM/Qwen#vllm) and [OpenAI interface usage](https://github.com/QwenLM/Qwen#openai-api) in our GitHub repo.

如果使用2xA100-80G进行部署，可以运行以下代码：

If deploying with 2xA100-80G, you can run the following code:

```python
python -m fastchat.serve.controller
python -m fastchat.serve.vllm_worker --model-path Qwen/Qwen-72B-Chat --trust-remote-code --tensor-parallel-size 2 --gpu-memory-utilization 0.98 --dtype bfloat16
# python -m fastchat.serve.vllm_worker --model-path Qwen/Qwen-72B-Chat-Int4 --trust-remote-code --dtype float16  # 运行int4模型。 run int4 model.
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

注意需要``--gpu-memory-utilization 0.98``参数避免OOM问题。

Note that the ``--gpu-memory-utilization 0.98`` parameter is required to avoid OOM problems.

<br>


关于更多的使用说明，请参考我们的[GitHub repo](https://github.com/QwenLM/Qwen)获取更多信息。

For more information, please refer to our [GitHub repo](https://github.com/QwenLM/Qwen) for more information.
<br>


## 量化 (Quantization)

### 用法 (Usage)

以下我们提供示例说明如何使用Int4/Int8量化模型。在开始使用前，请先保证满足要求（如torch 2.0及以上，transformers版本为4.32.0及以上，等等），并安装所需安装包：

Here we demonstrate how to use our provided quantized models for inference. Before you start, make sure you meet the requirements of auto-gptq (e.g., torch 2.0 and above, transformers 4.32.0 and above, etc.) and install the required packages:

```bash
pip install auto-gptq optimum
```

如安装`auto-gptq`遇到问题，我们建议您到官方[repo](https://github.com/PanQiWei/AutoGPTQ)搜索合适的预编译wheel。

If you meet problems installing `auto-gptq`, we advise you to check out the official [repo](https://github.com/PanQiWei/AutoGPTQ) to find a pre-build wheel.

> 注意：预编译的`auto-gptq`版本对`torch`版本及其CUDA版本要求严格。同时，由于
> 其近期更新，你可能会遇到`transformers`、`optimum`或`peft`抛出的版本错误。
> 我们建议使用符合以下要求的最新版本：
> - torch==2.1 auto-gptq>=0.5.1 transformers>=4.35.0 optimum>=1.14.0 peft>=0.6.1
> - torch>=2.0,<2.1 auto-gptq<0.5.0 transformers<4.35.0 optimum<1.14.0 peft>=0.5.0,<0.6.0
> Note: The pre-compiled `auto-gptq` packages strongly depend on the version of `torch` and its CUDA version. Moreover, due to recent update, 
> you may also encounter unsupported version errors from `transformers`, `optimum`, or `peft`.
> We recommend using the latest versions meeting the following requirements :
> - torch==2.1 auto-gptq>=0.5.1 transformers>=4.35.0 optimum>=1.14.0 peft>=0.6.1
> - torch>=2.0,<2.1 auto-gptq<0.5.0 transformers<4.35.0 optimum<1.14.0 peft>=0.5.0,<0.6.0

随后即可使用和上述一致的用法调用量化模型：

Then you can load the quantized model easily and run inference as same as usual:

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-72B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
```

注意：使用vLLM运行量化模型需安装我们[vLLM分支仓库](https://github.com/QwenLM/vllm-gptq)。暂不支持int8模型，近期将更新。

Note: You need to install our [vLLM repo] (https://github.com/qwenlm/vllm-gptq) for AutoGPTQ. The int8 model is not supported for the time being, and we will add the support soon.

### 效果评测

我们对BF16，Int8和Int4模型在基准评测上做了测试（使用zero-shot设置），结果如下所示：

We illustrate the zero-shot performance of both BF16, Int8 and Int4 models on the benchmark. Results are shown below:

| Quantization | MMLU | CEval (val) | GSM8K | Humaneval |
|--------------|:----:|:-----------:|:-----:|:---------:|
| BF16         | 74.4 |    80.1     | 76.4  |   64.6    |
| Int8         | 73.5 |    80.1     | 73.5  |   62.2    |
| Int4         | 73.4 |    80.1     | 75.3  |   61.6    |

### 推理速度及显存使用 (Inference Speed & GPU Memory Usage)

我们测算了不同精度模型、不同FlashAttn库版本、以及是否使用vLLM的情况下，模型在不同输入长度下生成2048词的平均推理速度以及显存使用。

We measured the average inference speed and GPU memory usage of generating 2048 tokens across several settings, including input lengths, quantization levels, versions of flash-attention, and whether vLLM is used.

|  Quantization |     Setting       | # of A100-80G GPUs |  Context Length | Generation Length | Speed (Tokens/s) | Total GPU Memory Usage | 
| ------------- | :---------------: | :----------------: | :-------------: | :---------------: | :---------------:| :---------------------:|
|      BF16     | HF + FlashAttn-v2 |        2           |       1         |       2048        |       8.48       |        144.69GB        |
|      BF16     | HF + FlashAttn-v1 |        2           |       1         |       2048        |       8.31       |        144.69GB        |
|      BF16     | HF + No FlashAttn |        2           |       1         |       2048        |       7.89       |        144.69GB        |
|      BF16     |       vLLM        |        2           |       1         |       2048        |      17.60       |      Pre-Allocated*    |
|      BF16     |       vLLM        |        4           |       1         |       2048        |      26.16       |      Pre-Allocated*    |
|      BF16     | HF + FlashAttn-v2 |        4           |      6144       |       2048        |       5.37       |        181.47GB        |
|      BF16     | HF + FlashAttn-v1 |        4           |      6144       |       2048        |       4.96       |        181.47GB        |
|      BF16     | HF + No FlashAttn |        4           |      6144       |       2048        |       4.72       |        202.74GB        |
|      BF16     |       vLLM        |        4           |      6144       |       2048        |      24.41       |      Pre-Allocated*    |
|      BF16     |       vLLM        |        4           |     14336       |       2048        |      21.24       |      Pre-Allocated*    |
|      BF16     |       vLLM        |        4           |     30720       |       2048        |      17.55       |      Pre-Allocated*    |
|      Int8     | HF + FlashAttn-v2 |        2           |       1         |       2048        |       9.05       |         81.27GB        |
|      Int8     | HF + FlashAttn-v1 |        2           |       1         |       2048        |       8.97       |         81.27GB        |
|      Int8     | HF + No FlashAttn |        2           |       1         |       2048        |       8.32       |         81.27GB        |
|      Int8     | HF + FlashAttn-v2 |        3           |      6144       |       2048        |       5.76       |        118.06GB        |
|      Int8     | HF + FlashAttn-v1 |        3           |      6144       |       2048        |       5.72       |        118.06GB        |
|      Int8     | HF + No FlashAttn |        2           |      6144       |       2048        |       4.50       |        129.83GB        |
|      Int8     | HF + FlashAttn-v2 |        4           |     14336       |       2048        |       3.44       |        180.44GB        |
|      Int8     | HF + FlashAttn-v1 |        4           |     14336       |       2048        |       3.19       |        180.44GB        |
|      Int8     | HF + No FlashAttn |        4           |     14336       |       2048        |       OOM        |            OOM         |
|      Int4     | HF + FlashAttn-v2 |        1           |       1         |       2048        |      11.67       |         48.86GB        |
|      Int4     | HF + FlashAttn-v1 |        1           |       1         |       2048        |      11.27       |         48.86GB        |
|      Int4     | HF + No FlashAttn |        1           |       1         |       2048        |      11.32       |         48.86GB        |
|      Int4     |       vLLM        |        1           |       1         |       2048        |      14.63       |      Pre-Allocated*    |
|      Int4     |       vLLM        |        2           |       1         |       2048        |      20.76       |      Pre-Allocated*    |
|      Int4     |       vLLM        |        4           |       1         |       2048        |      27.19       |      Pre-Allocated*    |
|      Int4     | HF + FlashAttn-v2 |        2           |      6144       |       2048        |       6.75       |         85.99GB        |
|      Int4     | HF + FlashAttn-v1 |        2           |      6144       |       2048        |       6.32       |         85.99GB        |
|      Int4     | HF + No FlashAttn |        2           |      6144       |       2048        |       5.97       |         88.30GB        |
|      Int4     |       vLLM        |        2           |      6144       |       2048        |      18.07       |      Pre-Allocated*    |
|      Int4     |       vLLM        |        4           |      6144       |       2048        |      24.56       |      Pre-Allocated*    |
|      Int4     | HF + FlashAttn-v2 |        3           |     14336       |       2048        |       4.18       |        148.73GB        |
|      Int4     | HF + FlashAttn-v1 |        3           |     14336       |       2048        |       3.72       |        148.73GB        |
|      Int4     | HF + No FlashAttn |        3           |     14336       |       2048        |       OOM        |            OOM         |
|      Int4     |       vLLM        |        2           |     14336       |       2048        |     	14.51       |      Pre-Allocated*    |
|      Int4     |       vLLM        |        4           |     14336       |       2048        |      19.28       |      Pre-Allocated*    |
|      Int4     |       vLLM        |        4           |     30720       |       2048        |      16.93       |      Pre-Allocated*    |

\* vLLM会提前预分配显存，因此无法探测最大显存使用情况。HF是指使用Huggingface Transformers库进行推理。

\* vLLM pre-allocates GPU memory, so we cannot detect the maximum usage. HF refers to using the Huggingface Transformers library for inference.

HuggingFace Transformers的性能测算使用[此脚本](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py)完成。评测使用A100-SXM4-80G GPU，使用PyTorch 2.0.1 (Huggingface Transformers) / PyTorch 2.1.0 (vLLM)和CUDA 11.8。

The speed and memory profiling of HuggingFace Transformers are conducted using [this script](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py). The profiling runs on A100-SXM4-80G GPUs with PyTorch 2.0.1 (for Huggingface Transformers) / PyTorch 2.1.0 (for vLLM) and CUDA 11.8. 
<br>

## 模型细节（Model）

与Qwen-72B预训练模型相同，Qwen-72B-Chat模型规模基本情况如下所示

The details of the model architecture of Qwen-72B-Chat are listed as follows

| Hyperparameter  |  Value |
|:----------------|:-------|
|    n_layers     |     80 |
|     n_heads     |     64 |
|     d_model     |   8192 |
|   vocab size    | 151851 |
| sequence length |  32768 |

在位置编码、FFN激活函数和normalization的实现方式上，我们也采用了目前最流行的做法，
即RoPE相对位置编码、SwiGLU激活函数、RMSNorm（可选安装flash-attention加速）。

在分词器方面，相比目前主流开源模型以中英词表为主，Qwen-72B-Chat使用了约15万token大小的词表。
该词表在GPT-4使用的BPE词表`cl100k_base`基础上，对中文、多语言进行了优化，在对中、英、代码数据的高效编解码的基础上，对部分多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强。
词表对数字按单个数字位切分。调用较为高效的[tiktoken分词库](https://github.com/openai/tiktoken)进行分词。

For position encoding, FFN activation function, and normalization calculation methods, we adopt the prevalent practices, i.e., RoPE relative position encoding, SwiGLU for activation function, and RMSNorm for normalization (optional installation of flash-attention for acceleration).

For tokenization, compared to the current mainstream open-source models based on Chinese and English vocabularies, Qwen-72B-Chat uses a vocabulary of over 150K tokens.
It first considers efficient encoding of Chinese, English, and code data, and is also more friendly to multilingual languages, enabling users to directly enhance the capability of some languages without expanding the vocabulary.
It segments numbers by single digit, and calls the [tiktoken](https://github.com/openai/tiktoken) tokenizer library for efficient tokenization.
<br>

## 评测效果（Evaluation）

对于Qwen-72B-Chat模型，我们同样评测了常规的中文理解（C-Eval）、英文理解（MMLU）、代码（HumanEval）和数学（GSM8K）等权威任务，同时包含了长序列任务的评测结果。由于Qwen-72B-Chat模型经过对齐后，激发了较强的外部系统调用能力，我们还进行了工具使用能力方面的评测。

提示：由于硬件和框架造成的舍入误差，复现结果如有波动属于正常现象。

For Qwen-72B-Chat, we also evaluate the model on C-Eval, MMLU, HumanEval, GSM8K, etc., as well as the benchmark evaluation for long-context understanding, and tool usage.

Note: Due to rounding errors caused by hardware and framework, differences in reproduced results are possible.

### 中文评测（Chinese Evaluation）

#### C-Eval

在[C-Eval](https://arxiv.org/abs/2305.08322)验证集上，我们评价了Qwen-72B-Chat模型的0-shot & 5-shot准确率

We demonstrate the 0-shot & 5-shot accuracy of Qwen-72B-Chat on C-Eval validation set

|              Model               | Avg. Acc. |
|:--------------------------------:|:---------:|
|          LLaMA2-7B-Chat          |   31.9    |
|         LLaMA2-13B-Chat          |   36.2    |
|         LLaMA2-70B-Chat          |   44.3    |
|         ChatGPT3.5               |   52.5    |
|         ChatGPT4                 |   69.9    |
|      Yi-34B-Chat (0-shot)        |   77.0    |
|      Yi-34B-Chat (5-shot)        |   78.5    |
| Qwen-7B-Chat (original) (0-shot) |   54.2    |
|    **Qwen-7B-Chat (0-shot)**     |   59.7    |
|    **Qwen-7B-Chat (5-shot)**     |   59.3    |
|    **Qwen-14B-Chat (0-shot)**    |   69.8    |
|    **Qwen-14B-Chat (5-shot)**    |   71.7    |
|    **Qwen-72B-Chat (0-shot)**    |   80.1    |
|    **Qwen-72B-Chat (5-shot)**    |   82.9    |


C-Eval测试集上，Qwen-72B-Chat模型的zero-shot准确率结果如下：

The zero-shot accuracy of Qwen-72B-Chat on C-Eval testing set is provided below:

| Model                   |   Avg.   | STEM | Social Sciences | Humanities | Others |
| :---------------------- | :------: | :--: | :-------------: | :--------: | :----: |
| Qwen-7B-Chat (original) |   54.6   | 47.8 |      67.6       |    59.3    |  50.6  |
| **Qwen-7B-Chat**        |   58.6   | 53.3 |      72.1       |    62.8    |  52.0  |
| **Qwen-14B-Chat**       |   69.1   | 65.1 |      80.9       |    71.2    |  63.4  |
| **Qwen-72B-Chat**       |   79.5   | 74.5 |      89.1       |    81.2    |  78.1  |

### 英文评测（English Evaluation）

#### MMLU

[MMLU](https://arxiv.org/abs/2009.03300)评测集上，Qwen-7B-Chat模型的 0-shot & 5-shot 准确率如下，效果同样在同类对齐模型中同样表现较优。

The 0-shot & 5-shot accuracy of Qwen-72B-Chat on MMLU is provided below.
The performance of Qwen-72B-Chat still on the top between other human-aligned models with comparable size.

|              Model               | Avg. Acc. |
|:--------------------------------:|:---------:|
|         LLaMA2-7B-Chat           |   46.2    |
|         LLaMA2-13B-Chat          |   54.6    |
|         LLaMA2-70B-Chat          |   63.8    |
|      Yi-34B-Chat (0-shot)        |   67.6    |
|      Yi-34B-Chat (5-shot)        |   73.4    |
|         ChatGPT3.5               |   69.1    |
|         ChatGPT4                 |   83.0    |
| Qwen-7B-Chat (original) (0-shot) |   53.9    |
|    **Qwen-7B-Chat (0-shot)**     |   55.8    |
|    **Qwen-7B-Chat (5-shot)**     |   57.0    |
|    **Qwen-14B-Chat (0-shot)**    |   64.6    |
|    **Qwen-14B-Chat (5-shot)**    |   66.5    |
|    **Qwen-72B-Chat (0-shot)**    |   74.3    |
|    **Qwen-72B-Chat (5-shot)**    |   75.0    |

### 代码评测（Coding Evaluation）

Qwen-72B-Chat在[HumanEval](https://github.com/openai/human-eval)的zero-shot Pass@1效果如下

The zero-shot Pass@1 of Qwen-72B-Chat on [HumanEval](https://github.com/openai/human-eval) is demonstrated below

|          Model          |  Pass@1  |
|:-----------------------:|:--------:|
|     LLaMA2-7B-Chat      |   12.2   |
|     LLaMA2-13B-Chat     |   18.9   |
|     LLaMA2-70B-Chat     |   32.3   |
|       Yi-34B-Chat       |   33.5   |
|       ChatGPT3.5        |   73.2   |
|       ChatGPT4          |   86.6   |
| Qwen-7B-Chat (original) |   24.4   |
|    **Qwen-7B-Chat**     |   37.2   |
|    **Qwen-14B-Chat**    |   43.9   |
|    **Qwen-72B-Chat**    |   64.6   |

### 数学评测（Mathematics Evaluation）

在评测数学能力的[GSM8K](https://github.com/openai/grade-school-math)上，Qwen-72B-Chat的准确率结果如下

The accuracy of Qwen-72B-Chat on GSM8K is shown below

|              Model               |   Acc.   |
|:--------------------------------:|:--------:|
|          LLaMA2-7B-Chat          |   26.3   |
|         LLaMA2-13B-Chat          |   37.1   |
|         LLaMA2-70B-Chat          |   59.3   |
|           Yi-34B-Chat            |   71.6   |
|           ChatGPT3.5             |   73.2   |
|           ChatGPT4               |   91.4   |
| Qwen-7B-Chat (original) (0-shot) |   41.1   |
|    **Qwen-7B-Chat (0-shot)**     |   50.3   |
|    **Qwen-7B-Chat (8-shot)**     |   54.1   |
|    **Qwen-14B-Chat (0-shot)**    |   60.1   |
|    **Qwen-14B-Chat (8-shot)**    |   59.3   |
|    **Qwen-72B-Chat (0-shot)**    |   76.4   |
|    **Qwen-72B-Chat (8-shot)**    |   75.7   |

### 长序列评测（Long-Context Understanding）

Qwen-72B-Chat支持最长32k的上下文长度，在[L-Eval](https://arxiv.org/abs/2307.11088)客观题的评分结果如下：

Qwen-72B-Chat supports context lengths of up to 32k. The scores of [L-Eval](https://arxiv.org/abs/2307.11088) (closed-ended tasks) are as follows:

| Model             |  Average   |  Coursera  |    GSM     |   QuALITY  |    TOEFL   |   CodeU    |  SFcition  |
|:------------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ChatGPT-3.5-16k   |    60.73   | **63.51**  | **84.00**  |   61.38    |    78.43   | **12.22**  |    64.84   |
| **Qwen-72B-Chat** |  **62.30** |   58.13    |   76.00    | **77.22**  |  **86.24** |    6.66    |  **69.53** |


我们进一步进行了“大海捞针”实验（想法来自于[@Greg Kamradt](https://twitter.com/GregKamradt/status/1727018183608193393)），测试模型在不同长度的输入下，是否能检索到文章不同位置的信息，结果如下：

We conducted the "needle in a haystack" experiment (the idea came from [@Greg Kamradt](https://twitter.com/GregKamradt/status/1727018183608193393)) to test whether the model can retrieve information at different positions in the inputs of different lengths, the result is as follows:

![](assets/qwen_72b_needle_in_a_haystack.png)

以上结果说明，Qwen-72B-Chat可以能准确检索到32k以内的输入长度中放在各种位置的信息，证明了其具有优秀的长文本处理能力。

The above results show that Qwen-72B-Chat can accurately retrieve information placed in various positions within an input length of 32k, proving its excellent long text understanding capabilities.

## FAQ

如遇到问题，敬请查阅[FAQ](https://github.com/QwenLM/Qwen/blob/main/FAQ_zh.md)以及issue区，如仍无法解决再提交issue。

If you meet problems, please refer to [FAQ](https://github.com/QwenLM/Qwen/blob/main/FAQ.md) and the issues first to search a solution before you launch a new issue.
<br>

## 引用 (Citation)

如果你觉得我们的工作对你有帮助，欢迎引用！

If you find our work helpful, feel free to give us a cite.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
<br>

## 使用协议（License Agreement）

我们的代码和模型权重对学术研究完全开放，并支持商用。请查看[LICENSE](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)了解具体的开源协议细节。如需商用，欢迎填写[问卷](https://dashscope.console.aliyun.com/openModelApply/Qwen-72B-Chat)申请。

Our code and checkpoints are open to research purpose, and they are allowed for commercial purposes. Check [LICENSE](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) for more details about the license. If you have requirements for commercial use, please fill out the [form](https://dashscope.console.aliyun.com/openModelApply/Qwen-72B-Chat) to apply.
<br>

## 联系我们（Contact Us）

如果你想给我们的研发团队和产品团队留言，欢迎加入我们的微信群、钉钉群以及Discord！同时，也欢迎通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

If you are interested to leave a message to either our research team or product team, join our Discord or WeChat groups! Also, feel free to send an email to qianwen_opensource@alibabacloud.com.

