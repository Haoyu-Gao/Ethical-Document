---
language:
- fr
license: mit
tags:
- generated_from_trainer
metrics:
- precision
- recall
- accuracy
- f1
widget:
- text: Le président de la République appelle en outre les Français à faire le choix
    d'une "majorité stable et sérieuse pour les protéger face aux crises et pour agir
    pour l'avenir". "Je vois dans le projet de Jean-Luc Mélenchon ou de Madame Le
    Pen un projet de désordre et de soumission. Ils expliquent qu'il faut sortir de
    nos alliances, de l'Europe, et bâtir des alliances stratégiques avec la Russie.
    C'est la soumission à la Russie", assure-t-il.
- text: Top départ à l’ouverture des bureaux de vote. La Polynésie et les Français
    résidant à l'étranger, dont certains ont déjà pu voter en ligne, sont invités
    aux urnes ce week-end pour le premier tour des législatives, samedi 4 juin pour
    le continent américain et les Caraïbes, et dimanche 5 juin pour le reste du monde.
    En France métropolitaine, les premier et second tours auront lieu les 12 et 19
    juin.
- text: Le ministère a aussi indiqué que des missiles russes ont frappé un centre
    d'entraînement d'artillerie dans la région de Soumy où travaillaient des instructeurs
    étrangers. Il a jouté qu'une autre frappe avait détruit une position de "mercenaires
    étrangers" dans la région d'Odessa.
- text: 'Le malaise est profond et ressemble à une crise existentielle. Fait rarissime
    au Quai d’Orsay, six syndicats et un collectif de 500 jeunes diplomates du ministère
    des Affaires étrangères ont appelé à la grève, jeudi 2 juin, pour protester contre
    la réforme de la haute fonction publique qui, à terme, entraînera la disparition
    des deux corps historiques de la diplomatie française : celui de ministre plénipotentiaire
    (ambassadeur) et celui de conseiller des affaires étrangères.'
- text: Ils se font passer pour des recruteurs de Lockheed Martin ou du géant britannique
    de la défense et de l’aérospatial BAE Systems. Ces soi-disant chasseurs de tête
    font miroiter des perspectives lucratives de carrière et des postes à responsabilité.
    Mais ce n’est que du vent. En réalité, il s’agit de cyberespions nord-coréens
    cherchant à voler des secrets industriels de groupes de défense ou du secteur
    de l’aérospatial, révèle Eset, une société slovaque de sécurité informatique,
    dans un rapport publié mardi 31 mai.
model-index:
- name: camembert-keyword-extractor
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# camembert-keyword-extractor

This model is a fine-tuned version of [camembert-base](https://huggingface.co/camembert-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2199
- Precision: 0.6743
- Recall: 0.6979
- Accuracy: 0.9346
- F1: 0.6859

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 8
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Precision | Recall | Accuracy | F1     |
|:-------------:|:-----:|:-----:|:---------------:|:---------:|:------:|:--------:|:------:|
| 0.1747        | 1.0   | 1875  | 0.1780          | 0.5935    | 0.7116 | 0.9258   | 0.6472 |
| 0.1375        | 2.0   | 3750  | 0.1588          | 0.6505    | 0.7032 | 0.9334   | 0.6759 |
| 0.1147        | 3.0   | 5625  | 0.1727          | 0.6825    | 0.6689 | 0.9355   | 0.6756 |
| 0.0969        | 4.0   | 7500  | 0.1759          | 0.6886    | 0.6621 | 0.9350   | 0.6751 |
| 0.0837        | 5.0   | 9375  | 0.1967          | 0.6688    | 0.7112 | 0.9348   | 0.6893 |
| 0.0746        | 6.0   | 11250 | 0.2088          | 0.6646    | 0.7114 | 0.9334   | 0.6872 |
| 0.0666        | 7.0   | 13125 | 0.2169          | 0.6713    | 0.7054 | 0.9347   | 0.6879 |
| 0.0634        | 8.0   | 15000 | 0.2199          | 0.6743    | 0.6979 | 0.9346   | 0.6859 |


### Framework versions

- Transformers 4.19.2
- Pytorch 1.11.0+cu113
- Datasets 2.2.2
- Tokenizers 0.12.1
