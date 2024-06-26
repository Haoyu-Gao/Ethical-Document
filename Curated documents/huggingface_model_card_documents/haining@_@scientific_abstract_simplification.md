---
language:
- en
license: mit
tags:
- text2text generation
metrics:
- sacrebleu
- bert_score
- rouge
- meteor
- sari
- ari
- Automated Readability Index
inference:
  parameters:
    do_sample: true
    max_length: 512
    top_p: 0.9
    repetition_penalty: 1.0
task:
  name: scientific abstract simplification
  type: text2text generation
widget:
- text: 'summarize, simplify, and contextualize: The COVID-19 pandemic presented enormous
    data challenges in the United States. Policy makers, epidemiological modelers,
    and health researchers all require up-to-date data on the pandemic and relevant
    public behavior, ideally at fine spatial and temporal resolution. The COVIDcast
    API is our attempt to fill this need: Operational since April 2020, it provides
    open access to both traditional public health surveillance signals (cases, deaths,
    and hospitalizations) and many auxiliary indicators of COVID-19 activity, such
    as signals extracted from deidentified medical claims data, massive online surveys,
    cell phone mobility data, and internet search trends. These are available at a
    fine geographic resolution (mostly at the county level) and are updated daily.
    The COVIDcast API also tracks all revisions to historical data, allowing modelers
    to account for the frequent revisions and backfill that are common for many public
    health data sources. All of the data are available in a common format through
    the API and accompanying R and Python software packages. This paper describes
    the data sources and signals, and provides examples demonstrating that the auxiliary
    signals in the COVIDcast API present information relevant to tracking COVID activity,
    augmenting traditional public health reporting and empowering research and decision-making.'
  example_title: covid-api paper, from PNAS
- text: 'summarize, simplify, and contextualize: Potato mop-top virus (PMTV) is considered
    an emerging threat to potato production in the United States. PMTV is transmitted
    by a soil-borne protist, Spongospora subterranean. Rapid, accurate, and sensitive
    detection of PMTV in leaves and tubers is an essential component in PMTV management
    program. A rapid test that can be adapted to in-field, on-site testing with minimal
    sample manipulation could help in ensuring the sanitary status of the produce
    in situations such as certification programs and shipping point inspections. Toward
    that goal, a rapid and highly sensitive recombinase polymerase amplification (RPA)-based
    test was developed for PMTV detection in potato tubers. The test combines the
    convenience of RPA assay with a simple sample extraction procedure, making it
    amenable to rapid on-site diagnosis of PMTV. Furthermore, the assay was duplexed
    with a plant internal control to monitor sample extraction and RPA reaction performance.
    The method described could detect as little as 10 fg of PMTV RNA transcript in
    various potato tissues, the diagnostic limit of detection (LOQ) similar to that
    of traditional molecular methods.'
  example_title: potato paper, from PLOS ONE
- text: 'summarize, simplify, and contextualize: One of the most thrilling cultural
    experiences is to hear live symphony-orchestra music build up from a whispering
    passage to a monumental fortissimo. The impact of such a crescendo has been thought
    to depend only on the musicians’ skill, but here we show that interactions between
    the concert-hall acoustics and listeners’ hearing also play a major role in musical
    dynamics. These interactions contribute to the shoebox-type concert hall’s established
    success, but little prior research has been devoted to dynamic expression in this
    three-part transmission chain as a complete system. More forceful orchestral playing
    disproportionately excites high frequency harmonics more than those near the note’s
    fundamental. This effect results in not only more sound energy, but also a different
    tone color. The concert hall transmits this sound, and the room geometry defines
    from which directions acoustic reflections arrive at the listener. Binaural directional
    hearing emphasizes high frequencies more when sound arrives from the sides of
    the head rather than from the median plane. Simultaneously, these same frequencies
    are emphasized by higher orchestral-playing dynamics. When the room geometry provides
    reflections from these directions, the perceived dynamic range is enhanced. Current
    room-acoustic evaluation methods assume linear behavior and thus neglect this
    effect. The hypothesis presented here is that the auditory excitation by reflections
    is emphasized with an orchestra forte most in concert halls with strong lateral
    reflections. The enhanced dynamic range provides an explanation for the success
    of rectangularly shaped concert-hall geometry.'
  example_title: music paper, from PNAS
- text: 'summarize, simplify, and contextualize: Children in industrialized cultures
    typically succeed on Give-N, a test of counting ability, by age 4. On the other
    hand, counting appears to be learned much later in the Tsimane’, an indigenous
    group in the Bolivian Amazon. This study tests three hypotheses for what may cause
    this difference in timing: (a) Tsimane’ children may be shy in providing behavioral
    responses to number tasks, (b) Tsimane’ children may not memorize the verbal list
    of number words early in acquisition, and/or (c) home environments may not support
    mathematical learning in the same way as in US samples, leading Tsimane’ children
    to primarily acquire mathematics through formalized schooling. Our results suggest
    that most of our subjects are not inhibited by shyness in responding to experimental
    tasks. We also find that Tsimane’ children (N = 100, ages 4-11) learn the verbal
    list later than US children, but even upon acquiring this list, still take time
    to pass Give-N tasks. We find that performance in counting varies across tasks
    and is related to formal schooling. These results highlight the importance of
    formal education, including instruction in the count list, in learning the meanings
    of the number words.'
  example_title: given-n paper, from PLOS ONE
---


# TL;DR

Scientific Abstract Simplification (SAS) is a tool designed to rewrite complex scientific abstracts into simpler, more comprehensible versions. Our objective is to make scientific knowledge universally accessible. If you have already experimented with our baseline model (`sas_baseline`), you will find that the current model surpasses its predecessor in terms of all evaluation metrics. Feel free to test it via the Hosted Inference API to your right. Simply select one of the provided examples or input your own scientific abstract. Just ensure to precede your text with the instruction, "summarize, simplify, and contextualize: ", followed by a space. For local usage, refer to the [Usage](#Usage) section."

# Project Description


Open science has significantly reduced barriers to accessing scientific papers.
However, attainable research does not entail accessible knowledge.
Consequently, many individuals might prefer to rely on succinct social media narratives rather than endeavour to comprehend a scientific paper.
This preference is understandable as humans often favor narratives over dry, technical information. 
So, why not "translate" these intricate scientific abstracts into simpler, more accessible narratives? 
Several prestigious journals have already initiated steps towards enhancing accessibility. 
For instance, PNAS requires authors to submit Significance Statements understandable to an 'undergraduate-educated scientist', while Science includes an editor's abstract to provide a swift overview of the paper's salient points.

In this project, our objective is to employ AI to rewrite scientific abstracts into easily understandable scientific narratives.
To facilitate this, we have curated two new datasets: one containing PNAS abstract-significance pairs and the other encapsulating editor abstracts from Science.
We utilize a Transformer model (a variant known as Flan-T5) to fine-tune our model for the task of simplifying scientific abstracts.
Initially, the model is fine-tuned utilizing multiple discrete instructions by amalgamating four pertinent tasks in a challenge-proportional manner (a strategy we refer to as Multi-Instruction Pretuning).
Subsequently, we continue the fine-tuning process exclusively with the abstract-significance corpus. Our model can generate lay summaries that outperform models fine-tuned solely with the abstract-significance corpus and models fine-tuned with traditional task combinations.
We hope our work can foster a more comprehensive understanding of scientific research, enabling a larger audience to benefit from open science.


- **Model type:** Language model
- **Developed by:** 
  - PIs: Jason Clark and Hannah McKelvey, Montana State University
  - Fellow: Haining Wang, Indiana University Bloomington; Deanna Zarrillo, Drexel University
  - Collaborator: Zuoyu Tian, Indiana University Bloomington
  - [LEADING](https://cci.drexel.edu/mrc/leading/) Montana State University Library, Project "TL;DR it": Automating Article Synopses for Search Engine Optimization and Citizen Science
- **Language(s) (NLP):** English
- **License:** MIT
- **Parent Model:** [FLAN-T5-large](https://huggingface.co/google/flan-t5-large)


# Usage

Use the code below to get started with the model. Remember to prepend the `INSTRUCTION` for best performance.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
INSTRUCTION = "summarize, simplify, and contextualize: "
tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")
model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification")
input_text = "The COVID-19 pandemic presented enormous data challenges in the United States. Policy makers, epidemiological modelers, and health researchers all require up-to-date data on the pandemic and relevant public behavior, ideally at fine spatial and temporal resolution. The COVIDcast API is our attempt to fill this need: Operational since April 2020, it provides open access to both traditional public health surveillance signals (cases, deaths, and hospitalizations) and many auxiliary indicators of COVID-19 activity, such as signals extracted from deidentified medical claims data, massive online surveys, cell phone mobility data, and internet search trends. These are available at a fine geographic resolution (mostly at the county level) and are updated daily. The COVIDcast API also tracks all revisions to historical data, allowing modelers to account for the frequent revisions and backfill that are common for many public health data sources. All of the data are available in a common format through the API and accompanying R and Python software packages. This paper describes the data sources and signals, and provides examples demonstrating that the auxiliary signals in the COVIDcast API present information relevant to tracking COVID activity, augmenting traditional public health reporting and empowering research and decision-making."
encoding = tokenizer(INSTRUCTION + input_text, 
                     max_length=672, 
                     padding='max_length', 
                     truncation=True, 
                     return_tensors='pt')
decoded_ids = model.generate(input_ids=encoding['input_ids'],
                             attention_mask=encoding['attention_mask'], 
                             max_length=512, 
                             top_p=.9, 
                             do_sample=True)
print(tokenizer.decode(decoded_ids[0], skip_special_tokens=True))
```


# Training

## Data

| Corpus                           | # Training/Dev/Test Samples | # Training Tokens (source, target) | # Validation Tokens (source, target) | # Test Tokens (source, target) | Note                                                                                                                                   |
|----------------------------------|-----------------------------|------------------------------------|--------------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Scientific Abstract-Significance | 3,030/200/200               | 707,071, 375,433                   | 45,697, 24,901                       | 46,985, 24,426                 | -                                                                                                                                      |
| Editor Abstract                  | 732/91/92                   | 154,808, 194,721                   | 19,675, 24,421                       | 19,539, 24,332                 | -                                                                                                                                      |
| Wiki Auto                        | 28,364/1,000/1,000          | 18,239,990, 12,547,272             | 643,157, 444,034                     | 642549, 444,883                | We used the ACL version, adopted from Huggingface datasets. The validation and test samples are split from the corpus and kept frozen. |
| CNN/DailyMail                    | 287,113/13,368/11,490       | -                                  | -                                    | -                              | We used the 2.0 version, adopted from Huggingface datasets.                                                                            |


## Setup

We finetuned the base model (flan-t5-large) on multiple relevant tasks with standard language modeling loss. During training, the source text of each task is prepended with an task-specific instruction and mapped to the corresponding target text. For example, "simplify: " is added before a wiki text, and the whole text is fed into the model to line up with the corresponding simple wiki text. The tuning process has two steps.

| Task                               | Corpus                           | Instruction                                | Optimal samples |
|------------------------------------|----------------------------------|--------------------------------------------|-----------------|
| Scientific Abstract Simplification | Scientific Abstract-Significance | "summarize, simplify, and contextualize: " | 39,200          |
| Recontextualization                | Editor Abstract                  | "contextualize: "                          | 2,200           |
| Simplification                     | Wiki Auto                        | "simplify: "                               | 57,000          |
| Summarization                      | CNN/DailyMail                    | "summarize: "                              | 165,000         |
| Total                              | Challenge-proportional Mixing    | n/a                                        | 263,400         |


- Multi-instruction pretuning: In the stage, we first created a task mixture using "challenge-proportional mixing" method. In a separate pilot study, for each task, we finetuned it on a base model and observed the number of samples when validation loss starts to rise. We mixed the samples of each task proportional to its optimal number of samples. A corpus is exhausted before upsampling if the number of total samples is smaller than its optimal number. We finetune with the task mixture (263,400 samples) with the aforementioned template.

- fine-tuning: In this stage, we continued finetuning the checkpoint solely with the Scientific Abstract-Significance corpus till optimal validation loss was observed.

The multi-instruction tuning and the retuning took roughly 63 hours and 8 hours, respectively, on two NVIDIA RTX A5000 (24GB memory each) GPUs. We saved the checkpoint with the lowest validation loss for inference. We used the AdamW optimizer and a learning rate of 3e-5 with fully sharded data parallel strategy across training stages. The batch size equals to 1.

 
# Evaluation

The model is evaluated on the SAS test set using SacreBLEU, METEOR, BERTScore, ROUGE, SARI, and ARI.

## Metrics
<details>
  <summary> Click to expand </summary>
  - [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu): SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores. Inspired by Rico Sennrich’s multi-bleu-detok.perl, it produces the official WMT scores but works with plain text. It also knows all the standard test sets and handles downloading, processing, and tokenization for you.
  - [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore): BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.
  - [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge)-1/2/L: ROUGE is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.
  - [METEOR](https://huggingface.co/spaces/evaluate-metric/meteor): METEOR, an automatic metric for machine translation evaluation that is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings; furthermore, METEOR can be easily extended to include more advanced matching strategies. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.
  - [SARI](https://huggingface.co/spaces/evaluate-metric/sari): SARI is a metric used for evaluating automatic text simplification systems. The metric compares the predicted simplified sentences against the reference and the source sentences. It explicitly measures the goodness of words that are added, deleted and kept by the system. Sari = (F1_add + F1_keep + P_del) / 3 where F1_add: n-gram F1 score for add operation F1_keep: n-gram F1 score for keep operation P_del: n-gram precision score for delete operation n = 4, as in the original paper.
  - [The Automated Readability Index (ARI)](https://www.readabilityformulas.com/automated-readability-index.php): ARI is a readability test designed to assess the understandability of a text. Like other popular readability formulas, the ARI formula outputs a number which approximates the grade level needed to comprehend the text. For example, if the ARI outputs the number 10, this equates to a high school student, ages 15-16 years old; a number 3 means students in 3rd grade (ages 8-9 yrs. old) should be able to comprehend the text.
</details>

Implementations of SacreBLEU, BERT Score, ROUGE, METEOR, and SARI are from Huggingface [`evaluate`](https://pypi.org/project/evaluate/) v.0.3.0. ARI is from [`py-readability-metrics`](https://pypi.org/project/py-readability-metrics/) v.1.4.5.


## Results 

We tested our model on the SAS test set (200 samples). We generate 10 lay summaries based on each sample's abstract. During generation, we used top-p sampling with p=0.9. The mean performance is reported below.


| Metrics        | SAS     |
|----------------|---------|
| SacreBLEU↑     | 25.60   |
| BERT Score F1↑ | 90.14   |
| ROUGE-1↑      | 52.28   |
| ROUGE-2↑      | 29.61   |
| ROUGE-L↑      | 38.02   |
| METEOR↑        | 43.75   |
| SARI↑          | 51.96   |
| ARI↓           | 17.04   |
Note: 1. Some generated texts are too short (less than 100 words) to calcualte meaningful ARI. We therefore concatenated adjecent five texts and compute ARI for the 400 longer texts (instead of original 2,000 texts). 2. BERT score, ROUGE, and METEOR are multiplied by 100.


# Contact
Please [contact us](mailto:hw56@indiana.edu) for any questions or suggestions.


# Disclaimer

This model is designed to make scientific abstracts more accessible. Its outputs should not be relied upon for any purpose outside of this scope. There is no guarantee that the generated text accurately reflects the research it is based on. When making important decisions, it is recommended to seek the advice of human experts or consult the original papers. 

# Acknowledgement
This research is supported by the Institute of Museum and Library Services (IMLS) RE-246450-OLS-20.