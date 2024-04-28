---
language:
- en
license: apache-2.0
datasets:
- natural_instructions
- the_pile
- cot
- Muennighoff/P3
inference:
  parameters:
    max_new_tokens: 5
    temperature: 1.0
    top_k: 1
pipeline_tag: text-generation
widget:
- example_title: Sentiment Analysis
  text: 'The task is to label the post''s emotion as sadness, joy, love, anger, fear,
    or surprise.


    Input: I''m feeling quite sad and sorry for myself but ill snap out of it soon.

    Output: sadness


    Input: I am just feeling cranky and blue.

    Output: anger


    Input: I can have for a treat or if i am feeling festive.

    Output:'
- example_title: Country Currency
  text: 'Return the currency of the given country.


    Input: Switzerland

    Output: Swiss Franc


    Input: India

    Output:'
- example_title: Tweet Eval Hate
  text: 'Label whether the following tweet contains hate speech against either immigrants
    or women. Hate Speech (HS) is commonly defined as any communication that disparages
    a person or a group on the basis of some characteristic such as race, color, ethnicity,
    gender, sexual orientation, nationality, religion, or other characteristics.

    Possible labels:

    1. hate speech

    2. not hate speech


    Tweet: HOW REFRESHING! In South Korea, there is no such thing as ''political correctness"
    when it comes to dealing with Muslim refugee wannabes via @user

    Label: hate speech


    Tweet: New to Twitter-- any men on here know what the process is to get #verified?

    Label: not hate speech


    Tweet: Dont worry @user you are and will always be the most hysterical woman.

    Label:'
- example_title: Entity Recognition
  text: 'Extract all the names of people, places, and organizations from the following
    sentences.


    Sentence: Satya Nadella, the CEO of Microsoft, was visiting the Bahamas last May.

    Entities: Satya Nadella, Microsoft, Bahamas


    Sentence: Pacific Northwest cities include Seattle and Portland, which I have
    visited with Vikash.

    Entities:'
- example_title: Data Clearning
  text: 'Format the data into a CSV file:


    Input: Jane Doe jane.doe@gmail.com (520) 382 2435

    Output: Jane Doe,jane.doe@gmail.com,520-382-2435


    Input: Peter Lee (510) 333-2429 email: peter@yahoo.com

    Output:'
---

<h1 style="font-size: 42px">GPT-JT<h1/>


***<p style="font-size: 24px">Feel free to try out our [Online Demo](https://huggingface.co/spaces/togethercomputer/GPT-JT)!</p>***


# Model Summary

> With a new decentralized training algorithm, we fine-tuned GPT-J (6B) on 3.53 billion tokens, resulting in GPT-JT (6B), a model that outperforms many 100B+ parameter models on classification benchmarks.

We incorporated a collection of open techniques and datasets to build GPT-JT:
- GPT-JT is a fork of [EleutherAI](https://www.eleuther.ai)'s [GPT-J (6B)](https://huggingface.co/EleutherAI/gpt-j-6B);
- We used [UL2](https://github.com/google-research/google-research/tree/master/ul2)'s training objective, allowing the model to see bidirectional context of the prompt;
- The model was trained on a large collection of diverse data, including [Chain-of-Thought (CoT)](https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html), [Public Pool of Prompts (P3) dataset](https://huggingface.co/datasets/bigscience/P3), [Natural-Instructions (NI) dataset](https://github.com/allenai/natural-instructions).

With the help of techniques mentioned above, GPT-JT significantly improves the performance of classification tasks over the original GPT-J, and even outperforms most 100B+ parameter models!

# Quick Start

```python
from transformers import pipeline
pipe = pipeline(model='togethercomputer/GPT-JT-6B-v1')
pipe('''"I love this!" Is it positive? A:''')
```
or
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1")
```

# License

The weights of GPT-JT-6B-v1 are licensed under version 2.0 of the Apache License.

# Training Details

## UL2 Training Objective

We train GPT-JT using UL2 training objective [1][2].
The original GPT-J uses causal mask (as shown below left) for autoregressive generation. So for each token, it can only see its previous context.
In order to fully leverage the context information, we continue to train GPT-J with UL2 training objectives, and uses causal mask with prefix (as shown below right) -- using bidirectional attention for the prompt / input and causal attention for token generation.
Intuitively, being able to see context bidirectionally might improve downstream tasks that require this information.

$$ 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 & 1 
\end{bmatrix}

\begin{bmatrix}
1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 & 1 
\end{bmatrix}  
$$

Furthermore, we leverage a large collection of data, including [Natural-Instructions](https://github.com/allenai/natural-instructions), [P3](https://huggingface.co/datasets/Muennighoff/P3), [MMLU-COT](https://github.com/jasonwei20/flan-2/blob/main/mmlu-cot.json), and [the Pile](https://huggingface.co/datasets/the_pile)
Specifically, we first conduct training for 2.62 billion tokens using the UL2 loss on the Pile, followed by 0.92 billion tokens with a mixture of the above datasets: 5% of COT, 20% of P3, 20% of NI, and 55% of the Pile.

## Hyperparameters

We used AdamW with a learning rate of 1e-5 and global batch size of 64 (16 for each data parallel worker).
We used mix-precision training where the activation is in FP16 while the optimizer states are kept in FP32.
We use both data parallelism and pipeline parallelism to conduct training.
During training, we truncate the input sequence to 2048 tokens, and for input sequence that contains less than 2048 tokens, we concatenate multiple sequences into one long sequence to improve the data efficiency.

## Infrastructure

We used [the Together Research Computer](https://together.xyz/) to conduct training. 

# References

[1]: Tay, Yi, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, and Donald Metzler. "Unifying Language Learning Paradigms." arXiv preprint arXiv:2205.05131 (2022).

[2]: Tay, Yi, Jason Wei, Hyung Won Chung, Vinh Q. Tran, David R. So, Siamak Shakeri, Xavier Garcia et al. "Transcending scaling laws with 0.1% extra compute." arXiv preprint arXiv:2210.11399 (2022).