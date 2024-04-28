---
language: en
license: apache-2.0
library_name: transformers
tags:
- deberta-v3-base
- deberta-v3
- deberta
- text-classification
- nli
- natural-language-inference
- multitask
- multi-task
- pipeline
- extreme-multi-task
- extreme-mtl
- tasksource
- zero-shot
- rlhf
datasets:
- glue
- super_glue
- anli
- tasksource/babi_nli
- sick
- snli
- scitail
- OpenAssistant/oasst1
- universal_dependencies
- hans
- qbao775/PARARULE-Plus
- alisawuffles/WANLI
- metaeval/recast
- sileod/probability_words_nli
- joey234/nan-nli
- pietrolesci/nli_fever
- pietrolesci/breaking_nli
- pietrolesci/conj_nli
- pietrolesci/fracas
- pietrolesci/dialogue_nli
- pietrolesci/mpe
- pietrolesci/dnc
- pietrolesci/gpt3_nli
- pietrolesci/recast_white
- pietrolesci/joci
- martn-nguyen/contrast_nli
- pietrolesci/robust_nli
- pietrolesci/robust_nli_is_sd
- pietrolesci/robust_nli_li_ts
- pietrolesci/gen_debiased_nli
- pietrolesci/add_one_rte
- metaeval/imppres
- pietrolesci/glue_diagnostics
- hlgd
- PolyAI/banking77
- paws
- quora
- medical_questions_pairs
- conll2003
- nlpaueb/finer-139
- Anthropic/hh-rlhf
- Anthropic/model-written-evals
- truthful_qa
- nightingal3/fig-qa
- tasksource/bigbench
- blimp
- cos_e
- cosmos_qa
- dream
- openbookqa
- qasc
- quartz
- quail
- head_qa
- sciq
- social_i_qa
- wiki_hop
- wiqa
- piqa
- hellaswag
- pkavumba/balanced-copa
- 12ml/e-CARE
- art
- tasksource/mmlu
- winogrande
- codah
- ai2_arc
- definite_pronoun_resolution
- swag
- math_qa
- metaeval/utilitarianism
- mteb/amazon_counterfactual
- SetFit/insincere-questions
- SetFit/toxic_conversations
- turingbench/TuringBench
- trec
- tals/vitaminc
- hope_edi
- strombergnlp/rumoureval_2019
- ethos
- tweet_eval
- discovery
- pragmeval
- silicone
- lex_glue
- papluca/language-identification
- imdb
- rotten_tomatoes
- ag_news
- yelp_review_full
- financial_phrasebank
- poem_sentiment
- dbpedia_14
- amazon_polarity
- app_reviews
- hate_speech18
- sms_spam
- humicroedit
- snips_built_in_intents
- banking77
- hate_speech_offensive
- yahoo_answers_topics
- pacovaldez/stackoverflow-questions
- zapsdcn/hyperpartisan_news
- zapsdcn/sciie
- zapsdcn/citation_intent
- go_emotions
- allenai/scicite
- liar
- relbert/lexical_relation_classification
- metaeval/linguisticprobing
- tasksource/crowdflower
- metaeval/ethics
- emo
- google_wellformed_query
- tweets_hate_speech_detection
- has_part
- wnut_17
- ncbi_disease
- acronym_identification
- jnlpba
- species_800
- SpeedOfMagic/ontonotes_english
- blog_authorship_corpus
- launch/open_question_type
- health_fact
- commonsense_qa
- mc_taco
- ade_corpus_v2
- prajjwal1/discosense
- circa
- PiC/phrase_similarity
- copenlu/scientific-exaggeration-detection
- quarel
- mwong/fever-evidence-related
- numer_sense
- dynabench/dynasent
- raquiba/Sarcasm_News_Headline
- sem_eval_2010_task_8
- demo-org/auditor_review
- medmcqa
- aqua_rat
- RuyuanWan/Dynasent_Disagreement
- RuyuanWan/Politeness_Disagreement
- RuyuanWan/SBIC_Disagreement
- RuyuanWan/SChem_Disagreement
- RuyuanWan/Dilemmas_Disagreement
- lucasmccabe/logiqa
- wiki_qa
- metaeval/cycic_classification
- metaeval/cycic_multiplechoice
- metaeval/sts-companion
- metaeval/commonsense_qa_2.0
- metaeval/lingnli
- metaeval/monotonicity-entailment
- metaeval/arct
- metaeval/scinli
- metaeval/naturallogic
- onestop_qa
- demelin/moral_stories
- corypaik/prost
- aps/dynahate
- metaeval/syntactic-augmentation-nli
- metaeval/autotnli
- lasha-nlp/CONDAQA
- openai/webgpt_comparisons
- Dahoas/synthetic-instruct-gptj-pairwise
- metaeval/scruples
- metaeval/wouldyourather
- sileod/attempto-nli
- metaeval/defeasible-nli
- metaeval/help-nli
- metaeval/nli-veridicality-transitivity
- metaeval/natural-language-satisfiability
- metaeval/lonli
- tasksource/dadc-limit-nli
- ColumbiaNLP/FLUTE
- metaeval/strategy-qa
- openai/summarize_from_feedback
- tasksource/folio
- metaeval/tomi-nli
- metaeval/avicenna
- stanfordnlp/SHP
- GBaker/MedQA-USMLE-4-options-hf
- GBaker/MedQA-USMLE-4-options
- sileod/wikimedqa
- declare-lab/cicero
- amydeng2000/CREAK
- metaeval/mutual
- inverse-scaling/NeQA
- inverse-scaling/quote-repetition
- inverse-scaling/redefine-math
- tasksource/puzzte
- metaeval/implicatures
- race
- metaeval/spartqa-yn
- metaeval/spartqa-mchoice
- metaeval/temporal-nli
- metaeval/ScienceQA_text_only
- AndyChiang/cloth
- metaeval/logiqa-2.0-nli
- tasksource/oasst1_dense_flat
- metaeval/boolq-natural-perturbations
- metaeval/path-naturalness-prediction
- riddle_sense
- Jiangjie/ekar_english
- metaeval/implicit-hate-stg1
- metaeval/chaos-mnli-ambiguity
- IlyaGusev/headline_cause
- metaeval/race-c
- metaeval/equate
- metaeval/ambient
- AndyChiang/dgen
- metaeval/clcd-english
- civil_comments
- metaeval/acceptability-prediction
- maximedb/twentyquestions
- metaeval/counterfactually-augmented-snli
- tasksource/I2D2
- sileod/mindgames
- metaeval/counterfactually-augmented-imdb
- metaeval/cnli
- metaeval/reclor
- tasksource/oasst1_pairwise_rlhf_reward
- tasksource/zero-shot-label-nli
- webis/args_me
- webis/Touche23-ValueEval
- tasksource/starcon
- tasksource/ruletaker
- lighteval/lsat_qa
- tasksource/ConTRoL-nli
- tasksource/tracie
- tasksource/sherliic
- tasksource/sen-making
- tasksource/winowhy
- mediabiasgroup/mbib-base
- tasksource/robustLR
- CLUTRR/v1
- tasksource/logical-fallacy
- tasksource/parade
- tasksource/cladder
- tasksource/subjectivity
- tasksource/MOH
- tasksource/VUAC
- tasksource/TroFi
- sharc_modified
- tasksource/conceptrules_v2
- tasksource/disrpt
- conll2000
- DFKI-SLT/few-nerd
- tasksource/com2sense
- tasksource/scone
- tasksource/winodict
- tasksource/fool-me-twice
- tasksource/monli
- tasksource/corr2cause
- tasksource/apt
- zeroshot/twitter-financial-news-sentiment
- tasksource/icl-symbol-tuning-instruct
- tasksource/SpaceNLI
- sihaochen/propsegment
- HannahRoseKirk/HatemojiBuild
- tasksource/regset
- tasksource/babi_nli
- lmsys/chatbot_arena_conversations
metrics:
- accuracy
pipeline_tag: zero-shot-classification
model-index:
- name: deberta-v3-base-tasksource-nli
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: glue
      type: glue
      config: rte
      split: validation
    metrics:
    - type: accuracy
      value: 0.89
  - task:
      type: natural-language-inference
      name: Natural Language Inference
    dataset:
      name: anli-r3
      type: anli
      config: plain_text
      split: validation
    metrics:
    - type: accuracy
      value: 0.52
      name: Accuracy
---

# Model Card for DeBERTa-v3-base-tasksource-nli

This is [DeBERTa-v3-base](https://hf.co/microsoft/deberta-v3-base) fine-tuned with multi-task learning on 600 tasks of the [tasksource collection](https://github.com/sileod/tasksource/).
This checkpoint has strong zero-shot validation performance on many tasks (e.g. 70% on WNLI), and can be used for:
- Zero-shot entailment-based classification for arbitrary labels [ZS].
- Natural language inference [NLI]
- Hundreds of previous tasks with tasksource-adapters [TA].
- Further fine-tuning on a new task or tasksource task (classification, token classification or multiple-choice) [FT].

# [ZS] Zero-shot classification pipeline
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification",model="sileod/deberta-v3-base-tasksource-nli")

text = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(text, candidate_labels)
```
NLI training data of this model includes [label-nli](https://huggingface.co/datasets/tasksource/zero-shot-label-nli), a NLI dataset specially constructed to improve this kind of zero-shot classification.

# [NLI] Natural language inference pipeline

```python
from transformers import pipeline
nli_pipe = pipeline("text-classification",model="sileod/deberta-v3-base-tasksource-nli")
make_input = lambda x: [dict(text=prem,text_pair=hyp) for prem,hyp in x]
nli_pipe(make_input([('there is a cat','there is a black cat')])) #list of (premise,hypothesis)
# [{'label': 'neutral', 'score': 0.9952911138534546}]
```

# [TA] Tasksource-adapters: 1 line access to hundreds of tasks 

```python
# !pip install tasknet
import tasknet as tn
pipe = tn.load_pipeline('sileod/deberta-v3-base-tasksource-nli','glue/sst2') # works for 500+ tasksource tasks
pipe(['That movie was great !', 'Awful movie.'])
# [{'label': 'positive', 'score': 0.9956}, {'label': 'negative', 'score': 0.9967}]
```
The list of tasks is available in model config.json.
This is more efficient than ZS since it requires only one forward pass per example, but it is less flexible.


# [FT] Tasknet: 3 lines fine-tuning

```python
# !pip install tasknet
import tasknet as tn
hparams=dict(model_name='sileod/deberta-v3-base-tasksource-nli', learning_rate=2e-5)
model, trainer = tn.Model_Trainer([tn.AutoTask("glue/rte")], hparams)
trainer.train()
```

## Evaluation
This model ranked 1st among all models with the microsoft/deberta-v3-base architecture according to the IBM model recycling evaluation.
https://ibm.github.io/model-recycling/

### Software and training details

The model was trained on 600 tasks for 200k steps with a batch size of 384 and a peak learning rate of 2e-5. Training took 12 days on Nvidia A30 24GB gpu.
This is the shared model with the MNLI classifier on top. Each task had a specific CLS embedding, which is dropped 10% of the time to facilitate model use without it. All multiple-choice model used the same classification layers. For classification tasks, models shared weights if their labels matched.


https://github.com/sileod/tasksource/ \
https://github.com/sileod/tasknet/ \
Training code: https://colab.research.google.com/drive/1iB4Oxl9_B5W3ZDzXoWJN-olUbqLBxgQS?usp=sharing

# Citation

More details on this [article:](https://arxiv.org/abs/2301.05948) 
```
@article{sileo2023tasksource,
  title={tasksource: Structured Dataset Preprocessing Annotations for Frictionless Extreme Multi-Task Learning and Evaluation},
  author={Sileo, Damien},
  url= {https://arxiv.org/abs/2301.05948},
  journal={arXiv preprint arXiv:2301.05948},
  year={2023}
}
```


# Model Card Contact

damien.sileo@inria.fr


</details>