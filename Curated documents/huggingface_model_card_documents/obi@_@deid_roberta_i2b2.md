---
language:
- en
license: mit
tags:
- deidentification
- medical notes
- ehr
- phi
datasets:
- I2B2
metrics:
- F1
- Recall
- Precision
thumbnail: https://www.onebraveidea.org/wp-content/uploads/2019/07/OBI-Logo-Website.png
widget:
- text: 'Physician Discharge Summary Admit date: 10/12/1982 Discharge date: 10/22/1982
    Patient Information Jack Reacher, 54 y.o. male (DOB = 1/21/1928).'
- text: 'Home Address: 123 Park Drive, San Diego, CA, 03245. Home Phone: 202-555-0199
    (home).'
- text: 'Hospital Care Team Service: Orthopedics Inpatient Attending: Roger C Kelly,
    MD Attending phys phone: (634)743-5135 Discharge Unit: HCS843 Primary Care Physician:
    Hassan V Kim, MD 512-832-5025.'
---

# Model Description

* A RoBERTa [[Liu et al., 2019]](https://arxiv.org/pdf/1907.11692.pdf) model fine-tuned for de-identification of medical notes.
* Sequence Labeling (token classification): The model was trained to predict protected health information (PHI/PII) entities (spans). A list of protected health information categories is given by [HIPAA](https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html).
* A token can either be classified as non-PHI or as one of the 11 PHI types. Token predictions are aggregated to spans by making use of BILOU tagging.
* The PHI labels that were used for training and other details can be found here: [Annotation Guidelines](https://github.com/obi-ml-public/ehr_deidentification/blob/master/AnnotationGuidelines.md)
* More details on how to use this model, the format of data and other useful information is present in the GitHub repo: [Robust DeID](https://github.com/obi-ml-public/ehr_deidentification).


# How to use

* A demo on how the model works (using model predictions to de-identify a medical note) is on this space: [Medical-Note-Deidentification](https://huggingface.co/spaces/obi/Medical-Note-Deidentification).
* Steps on how this model can be used to run a forward pass can be found here: [Forward Pass](https://github.com/obi-ml-public/ehr_deidentification/tree/master/steps/forward_pass)
* In brief, the steps are:
    * Sentencize (the model aggregates the sentences back to the note level) and tokenize the dataset.
    * Use the predict function of this model to gather the predictions (i.e., predictions for each token).
    * Additionally, the model predictions can be used to remove PHI from the original note/text.
    
    
# Dataset

* The I2B2 2014 [[Stubbs and Uzuner, 2015]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/) dataset was used to train this model.

|           | I2B2                  |            |  I2B2                |            |
| --------- | --------------------- | ---------- | -------------------- | ---------- |
|           | TRAIN SET - 790 NOTES |            | TEST SET - 514 NOTES |            |
| PHI LABEL | COUNT                 | PERCENTAGE | COUNT                | PERCENTAGE |
| DATE      | 7502                  | 43.69      | 4980                 | 44.14      |
| STAFF     | 3149                  | 18.34      | 2004                 | 17.76      |
| HOSP      | 1437                  | 8.37       | 875                  | 7.76       |
| AGE       | 1233                  | 7.18       | 764                  | 6.77       |
| LOC       | 1206                  | 7.02       | 856                  | 7.59       |
| PATIENT   | 1316                  | 7.66       | 879                  | 7.79       |
| PHONE     | 317                   | 1.85       | 217                  | 1.92       |
| ID        | 881                   | 5.13       | 625                  | 5.54       |
| PATORG    | 124                   | 0.72       | 82                   | 0.73       |
| EMAIL     | 4                     | 0.02       | 1                    | 0.01       |
| OTHERPHI  | 2                     | 0.01       | 0                    | 0          |
| TOTAL     | 17171                 | 100        | 11283                | 100        |


# Training procedure

* Steps on how this model was trained can be found here: [Training](https://github.com/obi-ml-public/ehr_deidentification/tree/master/steps/train). The "model_name_or_path" was set to: "roberta-large".
    * The dataset was sentencized with the en_core_sci_sm sentencizer from spacy.
    * The dataset was then tokenized with a custom tokenizer built on top of the en_core_sci_sm tokenizer from spacy.
    * For each sentence we added 32 tokens on the left (from previous sentences) and 32 tokens on the right (from the next sentences).
    * The added tokens are not used for learning - i.e, the loss is not computed on these tokens - they are used as additional context.
    * Each sequence contained a maximum of 128 tokens (including the 32 tokens added on). Longer sequences were split.
    * The sentencized and tokenized dataset with the token level labels based on the BILOU notation was used to train the model.
    * The model is fine-tuned from a pre-trained RoBERTa model.
    
* Training details:
    * Input sequence length: 128
    * Batch size: 32 (16 with 2 gradient accumulation steps)
    * Optimizer: AdamW 
    * Learning rate: 5e-5
    * Dropout: 0.1


## Results

# Questions?

Post a Github issue on the repo: [Robust DeID](https://github.com/obi-ml-public/ehr_deidentification).
