# XGLM multilingual model
## Version 1.0.0

### Model developer
Meta AI

### Model type
A family of multilingual autoregressive language models (ranging from 564 million to 7.5 billion parameters) trained on a balanced corpus of a diverse set of languages. The language model can learn tasks from natural language descriptions and a few examples.

### Model Feedback Channel
https://github.com/pytorch/fairseq

## Intended use
### Primary intended use
For research purposes only, e.g. reproducing model evaluation results. Generation is only used in a limited capacity for explanation/justification or for prompting/probing/priming for class labels.

### Out of scope uses
The primary purpose of the model is not to generate language, although the model is capable of doing that.

## Potential risks
This section lists the potential risks associated with using the model.

### Relevant factors
Based on known problems with NLP technology, potential relevant factors include output correctness, robustness, bias (gender, profession, race and religion), etc.

### Evaluation factors
The model was evaluated on hate speech detection and occupation identification.
* Hate speech detection (Huang et al. (2020)) - A safety task to test language models’ ability to identify hateful and offensive text.
* Occupation identification (De-Arteaga et al., 2019), (Zhao et al., 2020) - A bias task to study language models’ performance divergence between different gender groups on the task of occupation identification.

## Metrics
### Model performance measures
The XGLM model was primarily evaluated on
1. Zero shot and few shot learning by looking at per-language performance on tasks spanning commonsense reasoning (XCOPA, XWinograd), natural language inference (XNLI) and paraphrasing (PAWS-X). The model is also evaluated on XStoryCloze, a new dataset created by Meta AI.
2. Cross lingual transfer through templates and few-shot examples.
3. Knowledge probing - Evaluate to what extent the XGLM model can effectively store factual knowledge in different languages using the mLAMA benchmark.
4. Translation - We report machine translation results on WMT benchmarks and a subset of FLORES-101 in the main paper.

The model was also evaluated on hate speech datasets introduced by Huang et al. (2020) and an occupation identification dataset by De-Arteaga et al. 2019 to identify bias in the model.

### Approaches to handle uncertainty
Report confidence intervals, variance metrics for the model performance metrics. Few-shot evaluation was conducted with different sampling with 5 seeds. We reported statistical significance.

## Evaluation data
## Zero Shot and Few Shot evaluation

### XNLI (Conneau et al., 2018)
#### Description
The Cross-lingual Natural Language Inference (XNLI) corpus is the extension of the Multi-Genre NLI (MultiNLI) corpus to 15 languages. The dataset was created by manually translating the validation and test sets of MultiNLI into each of those 15 languages.

### XStoryCloze
#### Description
A new dataset created by Meta AI by translating the validation split of the English StoryCloze dataset (Mostafazadeh et al., 2016) (Spring 2016 version) to 10 other typologically diverse languages (ru, zh Simplified, es Latin America, ar, hi, id, te, sw, eu, my).

### XCOPA (Ponti et al., 2020)
#### Description
The Cross-lingual Choice of Plausible Alternatives (XCOPA) dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe.

### XWinograd (Tikhonov and Ryabinin, 2021)
#### Description
XWinograd is a multilingual collection of Winograd Schemas in six languages that can be used for evaluation of cross-lingual commonsense reasoning capabilities.

### PAWS-X (Yang et al., 2019)
#### Description
PAWS-X contains 23,659 human translated PAWS evaluation pairs and 296,406 machine translated training pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. All translated pairs are sourced from examples in PAWS-Wiki.

## Responsible AI (RAI) evaluation
### Hate speech (Huang et al. 2020)
This is a multilingual Twitter corpus for the task of hate speech detection with inferred four author demographic factors: age, country, gender and race/ethnicity. The corpus covers five languages: English, Italian, Polish, Portuguese and Spanish.

### Bias dataset (De-Arteaga et al. 2019)
The aim of this dataset is to study the gender bias of models that identify a person’s occupation from their bios.

----

## Training data
### CC100-XL
#### Description
Following the recent success of multilingual self-supervised pre-training (Devlin et al., 2019; Lample and Conneau, 2019; Con; Xue et al., 2020; Goyal et al., 2021a; Liu et al., 2020), we train our language models on a mixture of monolingual text of different languages. We extended the pipeline used for mining the CC100 corpus to generate CC100-XL, a significantly larger multilingual dataset covering 68 Common Crawl snapshots (from Summer 2013 to March/April 2020) and 134 languages.

More details on the CC100-XL dataset can be found in the Appendix section of the paper.

## RAI Dimensions
### Fairness (Bias and inclusion)
The XGLM model was evaluated on Hate speech and bias identification datasets. For hate speech, we observe that across the 5 languages in the dataset, in context learning results are only slightly better than random (50%). Another interesting observation is that most few shot results are worse than zero-shot, which indicates that the model is not able to utilize examples using the templates described in the paper. For bias identification, the XGLM (6.7B) English only model achieves the best performance on English and Spanish, while the GPT-3 model of comparable size (6.7B) model achieves the best in French. On certain occupations (e.g. model and teacher), XGLM 6.7B En only model and GPT-3 (6.7B) have very significant bias while XGLM 7.5B is much less biased.

### Privacy and security
The XGLM model did not have any special Privacy and Security considerations. The training data and evaluation data were both public and went through standard Meta AI Privacy and licensing procedures.

### Transparency and control
In the spirit of transparency and accountability we have created this model card and a data card for the CC100-XL which can be found in the Appendix section of the paper.

### Efficiency (Green AI)
From an engineering perspective, XGLM pertains to a family of models that represent single unified models catering to many languages which have wide application across many applications. Such a unified single model saves on carbon footprint as well as energy consumption (comparing to the alternative: separate models for different languages) leading to more energy efficiency. A single model, despite having the risk of being a single point of failure, has the powerful incentive of being easier to maintain, access, distribute, and track.
 
## References
Edoardo Maria Ponti, Goran Glavas, Olga Majewska, Qianchu Liu, Ivan Vulic, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 2362–2376. Association for Computational Linguistics.
XCOPA Dataset | Papers With Code

Alexey Tikhonov and Max Ryabinin. 2021. It’s all in the heads: Using attention heads as a baseline for cross-lingual transfer in commonsense reasoning. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 3534–3546. Association for Computational Linguistics. 
XWINO Dataset | Papers With Code (XWinograd)

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. 2019. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. CoRR, abs/1908.11828.
PAWS-X Dataset | Papers With Code

Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: evaluating cross-lingual sentence representations. CoRR, abs/1809.05053.
XNLI Dataset | Papers With Code 

Xiaolei Huang, Linzi Xing, Franck Dernoncourt, and Michael Paul. 2020. Multilingual twitter corpus and baselines for evaluating demographic bias in hate speech recognition. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 1440–1448. 

Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings of the Conference on Fairness, Accountability, and Transparency, pages 120–128.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, James F. Allen. A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories. CoRR abs/1604.01696.

Jieyu Zhao, Subhabrata Mukherjee, Saghar Hosseini, Kai-Wei Chang, and Ahmed Hassan Awadallah. 2020. Gender bias in multilingual embeddings and crosslingual transfer. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2896–2907.

## Citation details
```
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and
               Todor Mihaylov and
               Mikel Artetxe and
               Tianlu Wang and
               Shuohui Chen and
               Daniel Simig and
               Myle Ott and
               Naman Goyal and
               Shruti Bhosale and
               Jingfei Du and
               Ramakanth Pasunuru and
               Sam Shleifer and
               Punit Singh Koura and
               Vishrav Chaudhary and
               Brian O'Horo and
               Jeff Wang and
               Luke Zettlemoyer and
               Zornitsa Kozareva and
               Mona T. Diab and
               Veselin Stoyanov and
               Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668},
  eprinttype = {arXiv},
  eprint    = {2112.10668},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10668.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


# Model card for the paper ``Efficient Large Scale Language Modeling with Mixtures of Experts"
## Version 1.0.0

### Model developer
Meta AI

### Model type
An autoregressive English language model trained on a union of six English language models. We explore dense and sparse (MoE based) architectures in the paper.
* Dense models - Our dense models range from 125M parameters to 13B parameters.
* Sparse (MoE) models - Our MoE based models range from 15B parameters to 1.1 Trillion parameters.
This model card focuses on the 1.1 Trillion parameter model, but the discussion
applies to all of the models explored in this work.

### Citation details
Artetxe et al. (2021): Efficient Large Scale Language Modeling with Mixtures of Experts

### Model Feedback Channel
fairseq

## Intended use
### Primary intended use
For research purposes only, e.g. reproducing model evaluation results. Generation is only used in a limited capacity for explanation/justification or for prompting/probing/priming for class labels.

### Out of scope uses
The primary purpose of the model is not to generate language, although the model is capable of doing that.

## Factors influencing model performance
This section discusses potential risks associated with using the model.

### Relevant factors
Based on known problems with NLP technology, potential relevant factors include bias (gender, profession, race and religion).

### Evaluation factors
The 1.1T model was evaluated on StereoSet and CrowS-Pairs datasets to quantify encoded bias in the model.

## Metrics
### Model performance measures
The 1.1T parameter model was primarily evaluated on
1. In-domain and out-of-domain language modeling perplexity.
2. Zero-shot and few-shot priming.
3. Fully supervised finetuning.

### Approaches to handle uncertainty
For few-shot learning, we report the average results across 25 runs, randomly sampling a different set of few-shot examples from the training set each time.
 
## Evaluation data
## Zero Shot evaluation

### HellaSwag
#### Description
HellaSwag is a dataset for evaluating commonsense reasoning.

### PIQA
#### Description
PIQA is a dataset designed to evaluate reasoning about Physical Commonsense in Natural Language

### ReCoRd
#### Description
Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD) is a large-scale reading comprehension dataset which requires commonsense reasoning. ReCoRD consists of queries automatically generated from CNN/Daily Mail news articles; the answer to each query is a text span from a summarizing passage of the corresponding news. The goal of ReCoRD is to evaluate a machine's ability of commonsense reasoning in reading comprehension.

## Few Shot evaluation
### Winogrande
#### Description
Winogrande is a benchmark for commonsense reasoning. The dataset contains pronoun resolution problems originally designed to be unsolvable for statistical models that rely on selectional preferences or word associations.

### StoryCloze
#### Description
StoryCloze is a new commonsense reasoning framework for evaluating story understanding, story generation, and script learning. This test requires a system to choose the correct ending to a four-sentence story.

### OpenBookQA
#### Description
OpenBookQA is a new kind of question-answering dataset modeled after open book exams for assessing human understanding of a subject. It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe the understanding of a small “book” of 1,326 core science facts and the application of these facts to novel situations.

## Fully supervised evaluation

### BoolQ
#### Description
BoolQ is a question answering dataset for yes/no questions containing 15942 examples. These questions are naturally occurring – they are generated in unprompted and unconstrained settings. Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.

### SST-2
#### Description
SST-2 (or SST-binary) is a binary classification dataset where the goal is to differentiate between negative or somewhat negative vs somewhat positive or positive.

### MNLI
#### Description
The Multi-Genre Natural Language Inference (MultiNLI) corpus is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information. The corpus is modeled on the SNLI corpus, but differs in that covers a range of genres of spoken and written text, and supports a distinctive cross-genre generalization evaluation.

## Responsible AI (RAI) evaluation
### StereoSet
#### Description
A large-scale natural dataset in English to measure stereotypical biases in four domains: gender, profession, race, and religion

#### Motivation for dataset use
The motivation for evaluating the 1.1T parameter model on this dataset is to evaluate the model's stereotype bias in gender, profession, race, and religion

### CrowS
#### Description
Challenge Dataset for Measuring Social Biases in Masked Language Models

#### Motivation for dataset use
The motivation for evaluating the 1.1T parameter model on this dataset is to evaluate the model’s bias in the domains of race, religion and age

----

## Training data
### BookCorpus
#### Description
A dataset consisting of more than 10K unpublished books. 4GB in size. (Zhu et al., 2019)

### English Wikipedia
#### Description
Data from English wikipedia, excluding lists, tables and headers. 12GB in size.

### CC-News
#### Description
A dataset containing 63 millions English news articles crawled between September 2016 and February 2019. 76GB in size. (Nagel,2016)

### OpenWebText
#### Description
An open source recreation of the WebText dataset used to train GPT-2. 38GB in size. (Gokaslan and Cohen, 2019)

### CC-Stories
#### Description
A dataset containing a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas. 31GB in size. (Trinh and Le, 2018)

### English CC100
#### Description
A dataset extracted from CommonCrawl snapshots between January 2018 and December 2018, filtered to match the style of Wikipedia following the methodology introduced in CCNet (https://arxiv.org/abs/1911.00359). 292GB in size. (Wenzek et al., 2020)

## Responsible AI (RAI) Dimensions
### Fairness (Bias and inclusion)
The 1.1T parameter model was evaluated on the StereoSet and CrowS pairs dataset for inherent bias in the model, and bias as a result of the data. Similar to StereoSet, we observe that both the dense and MoE models get worse in terms of the Stereotype Score (SS) with scale.

### Privacy and security
The 1.1T model did not have any special Privacy and Security considerations. The training data and evaluation data were both public and went through standard Meta AI Privacy and licensing procedures.

### Transparency and control
In the spirit of transparency and accountability we have created this model card for the 1.1T parameter model and a data card for the training data (referenced in Artetxe et al. (2021)).

### Efficiency (Green AI)
The 1.1T parameter model is trained as a Mixture of Experts (MoE) model. Mixture of expert (MoE) models are efficient because they leverage sparse computation, i.e., only a small fraction of parameters are active for any given input. For instance, our 1.1T parameter MoE model requires only 30% more FLOPS compared to a 6.7B parameter dense model, i.e., a 160x increase in parameters with only a 30% increase in FLOPS. Notably, MoE models achieve much better validation perplexity for a given compute budget compared to dense models.

## References
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791– 4800, Florence, Italy. Association for Computational Linguistics.

Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):7432–7439.

Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme. 2018. ReCoRD: Bridging the gap between human and machine commonsense reading comprehension. arXiv preprint 1810.12885.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. Winogrande: An adversarial winograd schema challenge at scale. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8732–8740.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 839–849, San Diego, California. Association for Computational Linguistics.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.

Christopher Clark and Kenton Lee and Ming-Wei Chang and Tom Kwiatkowski and Michael Collins and Kristina Toutanova. 2019. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions

Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in pretrained language models. In Association for Computational Linguistics (ACL).

Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. 2020. CrowS-pairs: A challenge dataset for measuring social biases in masked language models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1953–1967, Online. Association for Computational Linguistics.

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2019. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. arXiv:1506.06724.

Sebastian Nagel. 2016. Cc-news. http: //web.archive.org/save/http: //commoncrawl.org/2016/10/news-dataset-available.

Aaron Gokaslan and Vanya Cohen. 2019. Openwebtext corpus. http://web.archive.org/save/http://Skylion007.github.io/OpenWebTextCorpus

Trieu H Trinh and Quoc V Le. 2018. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. 2020. CCNet: Extracting high quality monolingual datasets from web crawl data. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 4003–4012, Marseille, France. European Language Resources Association.