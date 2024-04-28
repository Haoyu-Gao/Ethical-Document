---
language: en
license: bsd-3-clause
pipeline_tag: text-generation
---

# ctrl

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Environmental Impact](#environmental-impact)
7. [Technical Specifications](#technical-specifications)
8. [Citation](#citation)
9. [Model Card Authors](#model-card-authors)
10. [How To Get Started With the Model](#how-to-get-started-with-the-model)


# Model Details

## Model Description

The CTRL model was proposed in [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher. It's a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.). The model developers released a model card for CTRL, available [here](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf).

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write: 

> The CTRL Language Model analyzed in this card generates text conditioned on control codes that specify domain, style, topics, dates, entities, relationships between entities, plot points, and task-related behavior.

- **Developed by:** See [associated paper](https://arxiv.org/abs/1909.05858) from Salesforce Research
- **Model type:** Transformer-based language model
- **Language(s) (NLP):** Primarily English, some German, Spanish, French
- **License:** [BSD 3-Clause](https://github.com/salesforce/ctrl/blob/master/LICENSE.txt); also see [Code of Conduct](https://github.com/salesforce/ctrl)
- **Related Models:** More information needed
    - **Parent Model:** More information needed
- **Resources for more information:** 
  - [Associated paper](https://arxiv.org/abs/1909.05858)
  - [GitHub repo](https://github.com/salesforce/ctrl)
  - [Developer Model Card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf)
  - [Blog post](https://blog.salesforceairesearch.com/introducing-a-conditional-transformer-language-model-for-controllable-generation/)

# Uses

## Direct Use

The model is a language model. The model can be used for text generation. 

## Downstream Use

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write that the primary intended users are general audiences and NLP Researchers, and that the primary intended uses are:

> 1. Generating artificial text in collaboration with a human, including but not limited to:
>   - Creative writing
>   - Automating repetitive writing tasks
>   - Formatting specific text types
>   - Creating contextualized marketing materials
> 2. Improvement of other NLP applications through fine-tuning (on another task or other data, e.g. fine-tuning CTRL to learn new kinds of language like product descriptions)
> 3. Enhancement in the field of natural language understanding to push towards a better understanding of artificial text generation, including how to detect it and work toward control, understanding, and potentially combating potentially negative consequences of such models.

## Out-of-Scope Use

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write: 

> - CTRL should not be used for generating artificial text without collaboration with a human.
> - It should not be used to make normative or prescriptive claims.
> - This software should not be used to promote or profit from:
>   - violence, hate, and division;
>   - environmental destruction;
>   - abuse of human rights; or
>   - the destruction of people's physical and mental health.

# Bias, Risks, and Limitations

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). Predictions generated by the model may include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups.

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write: 

> We recognize the potential for misuse or abuse, including use by bad actors who could manipulate the system to act maliciously and generate text to influence decision-making in political, economic, and social settings. False attribution could also harm individuals, organizations, or other entities. To address these concerns, the model was evaluated internally as well as externally by third parties, including the Partnership on AI, prior to release.

> To mitigate potential misuse to the extent possible, we stripped out all detectable training data from undesirable sources. We then redteamed the model and found that negative utterances were often placed in contexts that made them identifiable as such. For example, when using the ‘News’ control code, hate speech could be embedded as part of an apology (e.g. “the politician apologized for saying [insert hateful statement]”), implying that this type of speech was negative. By pre-selecting the available control codes (omitting, for example, Instagram and Twitter from the available domains), we are able to limit the potential for misuse.

> In releasing our model, we hope to put it into the hands of researchers and prosocial actors so that they can work to control, understand, and potentially combat the negative consequences of such models. We hope that research into detecting fake news and model-generated content of all kinds will be pushed forward by CTRL. It is our belief that these models should become a common tool so researchers can design methods to guard against malicious use and so the public becomes familiar with their existence and patterns of behavior.

See the [associated paper](https://arxiv.org/pdf/1909.05858.pdf) for further discussions about the ethics of LLMs.

## Recommendations

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write: 

> - A recommendation to monitor and detect use will be implemented through the development of a model that will identify CTRLgenerated text.
> - A second recommendation to further screen the input into and output from the model will be implemented through the addition of a check in the CTRL interface to prohibit the insertion into the model of certain negative inputs, which will help control the output that can be generated.
> - The model is trained on a limited number of languages: primarily English and some German, Spanish, French. A recommendation for a future area of research is to train the model on more languages.

See the [CTRL-detector GitHub repo](https://github.com/salesforce/ctrl-detector) for more on the detector model.

# Training

## Training Data

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write: 

> This model is trained on 140 GB of text drawn from a variety of domains: Wikipedia (English, German, Spanish, and French), Project Gutenberg, submissions from 45 subreddits, OpenWebText, a large collection of news data, Amazon Reviews, Europarl and UN data from WMT (En-De, En-Es, En-Fr), question-answer pairs (no context documents) from ELI5, and the MRQA shared task, which includes Stanford Question Answering Dataset, NewsQA, TriviaQA, SearchQA, HotpotQA, and Natural Questions. See the paper for the full list of training data.

## Training Procedure

### Preprocessing

In the [associated paper](https://arxiv.org/pdf/1909.05858.pdf) the developers write: 

> We learn BPE (Sennrich et al., 2015) codes and tokenize the data using fastBPE4, but we use a large vocabulary of roughly 250K tokens. This includes the sub-word tokens necessary to mitigate problems with rare words, but it also reduces the average number of tokens required to generate long text by including most common words. We use English Wikipedia and a 5% split of our collected OpenWebText data for learning BPE codes. We also introduce an unknown token so that during preprocessing we can filter out sequences that contain more than 2 unknown tokens. This, along with the compressed storage for efficient training (TFRecords) (Abadi et al., 2016), reduces our training data to 140 GB from the total 180 GB collected.

See the paper for links, references, and further details.

### Training

In the [associated paper](https://arxiv.org/pdf/1909.05858.pdf) the developers write: 

> CTRL has model dimension d = 1280, inner dimension f = 8192, 48 layers, and 16 heads per layer. Dropout with probability 0.1 follows the residual connections in each layer. Token embeddings were tied with the final output embedding layer (Inan et al., 2016; Press & Wolf, 2016).

See the paper for links, references, and further details.
 
# Evaluation

## Testing Data, Factors & Metrics

In their [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf), the developers write that model performance measures are: 

> Performance evaluated on qualitative judgments by humans as to whether the control codes lead to text generated in the desired domain

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). Details are pulled from the [associated paper](https://arxiv.org/pdf/1909.05858.pdf).

- **Hardware Type:** TPU v3 Pod
- **Hours used:** Approximately 336 hours (2 weeks)
- **Cloud Provider:** GCP
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Technical Specifications

In the [associated paper](https://arxiv.org/pdf/1909.05858.pdf) the developers write: 

> CTRL was implemented in TensorFlow (Abadi et al., 2016) and trained with a global batch size of 1024 distributed across 256 cores of a Cloud TPU v3 Pod for 800k iterations. Training took approximately 2 weeks using Adagrad (Duchi et al., 2011) with a linear warmup from 0 to 0.05 over 25k steps. The norm of gradients were clipped to 0.25 as in (Merity et al., 2017). Learning rate decay was not necessary due to the monotonic nature of the Adagrad accumulator. We compared to the Adam optimizer (Kingma & Ba, 2014) while training smaller models, but we noticed comparable convergence rates and significant memory savings with Adagrad. We also experimented with explicit memory-saving optimizers including SM3 (Anil et al., 2019), Adafactor (Shazeer & Stern, 2018), and NovoGrad (Ginsburg et al., 2019) with mixed results.

See the paper for links, references, and further details.

# Citation

**BibTeX:**

```bibtex
@article{keskarCTRL2019,
  title={{CTRL - A Conditional Transformer Language Model for Controllable Generation}},
  author={Keskar, Nitish Shirish and McCann, Bryan and Varshney, Lav and Xiong, Caiming and Socher, Richard},
  journal={arXiv preprint arXiv:1909.05858},
  year={2019}
}
```

**APA:**
- Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). Ctrl: A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05858.

# Model Card Authors

This model card was written by the team at Hugging Face, referencing the [model card](https://github.com/salesforce/ctrl/blob/master/ModelCard.pdf) released by the developers.

# How to Get Started with the Model

Use the code below to get started with the model. See the [Hugging Face ctrl docs](https://huggingface.co/docs/transformers/model_doc/ctrl) for more information.

<details>
<summary> Click to expand </summary>

```python
>>> from transformers import CTRLTokenizer, CTRLModel
>>> import torch

>>> tokenizer = CTRLTokenizer.from_pretrained("ctrl")
>>> model = CTRLModel.from_pretrained("ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
```

</details>