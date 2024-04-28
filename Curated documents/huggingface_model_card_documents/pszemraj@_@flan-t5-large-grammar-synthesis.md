---
license:
- cc-by-nc-sa-4.0
- apache-2.0
tags:
- grammar
- spelling
- punctuation
- error-correction
- grammar synthesis
- FLAN
datasets:
- jfleg
languages:
- en
widget:
- text: There car broke down so their hitching a ride to they're class.
  example_title: compound-1
- text: i can has cheezburger
  example_title: cheezburger
- text: so em if we have an now so with fito ringina know how to estimate the tren
    given the ereafte mylite trend we can also em an estimate is nod s i again tort
    watfettering an we have estimated the trend an called wot to be called sthat of
    exty right now we can and look at wy this should not hare a trend i becan we just
    remove the trend an and we can we now estimate tesees ona effect of them exty
  example_title: Transcribed Audio Example 2
- text: My coworker said he used a financial planner to help choose his stocks so
    he wouldn't loose money.
  example_title: incorrect word choice (context)
- text: good so hve on an tadley i'm not able to make it to the exla session on monday
    this week e which is why i am e recording pre recording an this excelleision and
    so to day i want e to talk about two things and first of all em i wont em wene
    give a summary er about ta ohow to remove trents in these nalitives from time
    series
  example_title: lowercased audio transcription output
- text: Frustrated, the chairs took me forever to set up.
  example_title: dangling modifier
- text: I would like a peice of pie.
  example_title: miss-spelling
- text: Which part of Zurich was you going to go hiking in when we were there for
    the first time together? ! ?
  example_title: chatbot on Zurich
- text: Most of the course is about semantic or  content of language but there are
    also interesting topics to be learned from the servicefeatures except statistics
    in characters in documents. At this point, Elvthos introduces himself as his native
    English speaker and goes on to say that if you continue to work on social scnce,
  example_title: social science ASR summary output
- text: they are somewhat nearby right yes please i'm not sure how the innish is tepen
    thut mayyouselect one that istatte lo variants in their property e ere interested
    and anyone basical e may be applyind reaching the browing approach were
  example_title: medical course audio transcription
parameters:
  max_length: 128
  min_length: 4
  num_beams: 8
  repetition_penalty: 1.21
  length_penalty: 1
  early_stopping: true
---


# grammar-synthesis-large: FLAN-t5

 <a href="https://colab.research.google.com/gist/pszemraj/5dc89199a631a9c6cfd7e386011452a0/demo-flan-t5-large-grammar-synthesis.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A fine-tuned version of [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) for grammar correction on an expanded version of the [JFLEG](https://paperswithcode.com/dataset/jfleg) dataset. [Demo](https://huggingface.co/spaces/pszemraj/FLAN-grammar-correction) on HF spaces.

## Example

![example](https://i.imgur.com/PIhrc7E.png)

Compare vs. the original [grammar-synthesis-large](https://huggingface.co/pszemraj/grammar-synthesis-large).

---

## usage in Python 

> There's a colab notebook that already has this basic version implemented (_click on the Open in Colab button_)

After `pip install transformers` run the following code:

```python
from transformers import pipeline

corrector = pipeline(
              'text2text-generation',
              'pszemraj/flan-t5-large-grammar-synthesis',
              )
raw_text = 'i can has cheezburger'
results = corrector(raw_text)
print(results)
```

**For Batch Inference:** see [this discussion thread](https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis/discussions/1) for details, but essentially the dataset consists of several sentences at a time, and so I'd recommend running inference **in the same fashion:** batches of 64-96 tokens ish (or, 2-3 sentences split with regex) 

- it is also helpful to **first** check whether or not a given sentence needs grammar correction before using the text2text model. You can do this with BERT-type models fine-tuned on CoLA like `textattack/roberta-base-CoLA`
- I made a notebook demonstrating batch inference [here](https://colab.research.google.com/gist/pszemraj/6e961b08970f98479511bb1e17cdb4f0/batch-grammar-check-correct-demo.ipynb)



---


## Model description

The intent is to create a text2text language model that successfully completes "single-shot grammar correction" on a potentially grammatically incorrect text **that could have a lot of mistakes** with the important qualifier of **it does not semantically change text/information that IS grammatically correct.**

Compare some of the heavier-error examples on [other grammar correction models](https://huggingface.co/models?dataset=dataset:jfleg) to see the difference :)

### ONNX Checkpoint

This model has been converted to ONNX and can be loaded/used with huggingface's `optimum` library.

You first need to [install optimum](https://huggingface.co/docs/optimum/installation)

```bash
pip install optimum[onnxruntime]
# ^ if you want to use a different runtime read their docs
```
load with the optimum `pipeline`

```python
from optimum.pipelines import pipeline

corrector = pipeline(
    "text2text-generation", model=corrector_model_name, accelerator="ort"
)
# use as normal
```

### Other checkpoints

If trading a slight decrease in grammatical correction quality for faster inference speed makes sense for your use case, check out the **[base](https://huggingface.co/pszemraj/grammar-synthesis-base)** and **[small](https://huggingface.co/pszemraj/grammar-synthesis-small)** checkpoints fine-tuned from the relevant t5 checkpoints. 

## Limitations

- dataset: `cc-by-nc-sa-4.0`
- model: `apache-2.0`
- this is **still a work-in-progress** and while probably useful for "single-shot grammar correction" in a lot of cases, **give the outputs a glance for correctness ok?**


## Use Cases

Obviously, this section is quite general as there are many things one can use "general single-shot grammar correction" for. Some ideas or use cases:

1. Correcting highly error-prone LM outputs. Some examples would be audio transcription (ASR) (this is literally some of the examples) or something like handwriting OCR. 
    - To be investigated further, depending on what model/system is used it _might_ be worth it to apply this after OCR on typed characters. 
2. Correcting/infilling text generated by text generation models to be cohesive/remove obvious errors that break the conversation immersion. I use this on the outputs of [this OPT 2.7B chatbot-esque model of myself](https://huggingface.co/pszemraj/opt-peter-2.7B).
  > An example of this model running on CPU with beam search:
  
```
Original response:
                ive heard it attributed to a bunch of different philosophical schools, including stoicism, pragmatism, existentialism and even some forms of post-structuralism. i think one of the most interesting (and most difficult) philosophical problems is trying to let dogs (or other animals) out of cages. the reason why this is a difficult problem is because it seems to go against our grain (so to
synthesizing took 306.12 seconds
Final response in 1294.857 s:
        I've heard it attributed to a bunch of different philosophical schools, including solipsism, pragmatism, existentialism and even some forms of post-structuralism. i think one of the most interesting (and most difficult) philosophical problems is trying to let dogs (or other animals) out of cages. the reason why this is a difficult problem is because it seems to go against our grain (so to speak)
```
  _Note: that I have some other logic that removes any periods at the end of the final sentence in this chatbot setting [to avoid coming off as passive aggressive](https://www.npr.org/2020/09/05/909969004/before-texting-your-kid-make-sure-to-double-check-your-punctuation)_
  
3. Somewhat related to #2 above, fixing/correcting so-called [tortured-phrases](https://arxiv.org/abs/2107.06751) that are dead giveaways text was generated by a language model. _Note that _SOME_ of these are not fixed, especially as they venture into domain-specific terminology (i.e. irregular timberland instead of Random Forest)._

---

## Citation info

If you find this fine-tuned model useful in your work, please consider citing it :)

```
@misc {peter_szemraj_2022,
	author       = { {Peter Szemraj} },
	title        = { flan-t5-large-grammar-synthesis (Revision d0b5ae2) },
	year         = 2022,
	url          = { https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis },
	doi          = { 10.57967/hf/0138 },
	publisher    = { Hugging Face }
}
```