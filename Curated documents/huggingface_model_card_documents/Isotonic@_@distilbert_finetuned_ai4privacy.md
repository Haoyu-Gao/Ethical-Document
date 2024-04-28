---
language:
- en
library_name: transformers
tags:
- generated_from_trainer
datasets:
- ai4privacy/pii-masking-65k
- ai4privacy/pii-masking-43k
metrics:
- f1
- precision
- recall
base_model: distilbert-base-uncased
pipeline_tag: token-classification
model-index:
- name: distilbert_finetuned_ai4privacy
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert_finetuned_ai4privacy

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the english only section of ai4privacy/pii-masking-65k dataset.

Latest Model: [electra_large_finetuned_ai4privacy_50k](https://huggingface.co/Isotonic/electra_large_finetuned_ai4privacy_50k)

## Useage
GitHub Implementation: [Ai4Privacy](https://github.com/Sripaad/ai4privacy)

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.2
- num_epochs: 7

## Class wise metrics
It achieves the following results on the evaluation set:
- Loss: 0.0106
- Overall Precision: 0.9760
- Overall Recall: 0.9801
- Overall F1: 0.9780
- Overall Accuracy: 0.9977
- Accountname F1: 1.0
- Accountnumber F1: 1.0
- Amount F1: 0.9565
- Bic F1: 1.0
- Bitcoinaddress F1: 1.0
- Buildingnumber F1: 0.9753
- City F1: 0.9987
- Company Name F1: 1.0
- County F1: 1.0
- Creditcardcvv F1: 0.9701
- Creditcardissuer F1: 0.9939
- Creditcardnumber F1: 1.0
- Currency F1: 0.8668
- Currencycode F1: 0.8662
- Currencyname F1: 0.7582
- Currencysymbol F1: 0.36
- Date F1: 0.9944
- Displayname F1: 0.5970
- Email F1: 1.0
- Ethereumaddress F1: 1.0
- Firstname F1: 0.9493
- Fullname F1: 0.9982
- Gender F1: 0.9524
- Iban F1: 1.0
- Ip F1: 0.5543
- Ipv4 F1: 0.8700
- Ipv6 F1: 0.8863
- Jobarea F1: 0.9806
- Jobdescriptor F1: 0.6875
- Jobtitle F1: 0.9424
- Jobtype F1: 0.8811
- Lastname F1: 0.9052
- Litecoinaddress F1: 0.9848
- Mac F1: 1.0
- Maskednumber F1: 1.0
- Middlename F1: 0.7364
- Name F1: 0.9994
- Nearbygpscoordinate F1: 0.5
- Number F1: 1.0
- Password F1: 1.0
- Phoneimei F1: 1.0
- Phone Number F1: 1.0
- Pin F1: 0.9697
- Prefix F1: 0.9540
- Secondaryaddress F1: 0.9947
- Sex F1: 0.9650
- Sextype F1: 0.0
- Ssn F1: 1.0
- State F1: 0.9965
- Street F1: 0.9810
- Streetaddress F1: 0.9832
- Suffix F1: 0.7928
- Time F1: 0.9880
- Url F1: 0.9974
- Useragent F1: 1.0
- Username F1: 0.9746
- Vehiclevin F1: 1.0
- Vehiclevrm F1: 1.0
- Zipcode F1: 0.9969

## Training results

| Training Loss | Epoch | Step | Validation Loss | Overall Precision | Overall Recall | Overall F1 | Overall Accuracy | Accountname F1 | Accountnumber F1 | Amount F1 | Bic F1 | Bitcoinaddress F1 | Buildingnumber F1 | City F1 | Company Name F1 | County F1 | Creditcardcvv F1 | Creditcardissuer F1 | Creditcardnumber F1 | Currency F1 | Currencycode F1 | Currencyname F1 | Currencysymbol F1 | Date F1 | Displayname F1 | Email F1 | Ethereumaddress F1 | Firstname F1 | Fullname F1 | Gender F1 | Iban F1 | Ip F1  | Ipv4 F1 | Ipv6 F1 | Jobarea F1 | Jobdescriptor F1 | Jobtitle F1 | Jobtype F1 | Lastname F1 | Litecoinaddress F1 | Mac F1 | Maskednumber F1 | Middlename F1 | Name F1 | Nearbygpscoordinate F1 | Number F1 | Password F1 | Phoneimei F1 | Phone Number F1 | Pin F1 | Prefix F1 | Secondaryaddress F1 | Sex F1 | Sextype F1 | Ssn F1 | State F1 | Street F1 | Streetaddress F1 | Suffix F1 | Time F1 | Url F1 | Useragent F1 | Username F1 | Vehiclevin F1 | Vehiclevrm F1 | Zipcode F1 |
|:-------------:|:-----:|:----:|:---------------:|:-----------------:|:--------------:|:----------:|:----------------:|:--------------:|:----------------:|:---------:|:------:|:-----------------:|:-----------------:|:-------:|:---------------:|:---------:|:----------------:|:-------------------:|:-------------------:|:-----------:|:---------------:|:---------------:|:-----------------:|:-------:|:--------------:|:--------:|:------------------:|:------------:|:-----------:|:---------:|:-------:|:------:|:-------:|:-------:|:----------:|:----------------:|:-----------:|:----------:|:-----------:|:------------------:|:------:|:---------------:|:-------------:|:-------:|:----------------------:|:---------:|:-----------:|:------------:|:---------------:|:------:|:---------:|:-------------------:|:------:|:----------:|:------:|:--------:|:---------:|:----------------:|:---------:|:-------:|:------:|:------------:|:-----------:|:-------------:|:-------------:|:----------:|
| No log        | 1.0   | 335  | 0.3836          | 0.6166            | 0.6314         | 0.6239     | 0.9080           | 0.0            | 0.5534           | 0.1940    | 0.0    | 0.4890            | 0.0               | 0.6856  | 0.0             | 0.0       | 0.0              | 0.0                 | 0.0                 | 0.0         | 0.0             | 0.0             | 0.0               | 0.3306  | 0.0            | 0.9420   | 0.4869             | 0.0704       | 0.9094      | 0.0       | 0.0877  | 0.0    | 0.6112  | 0.6779  | 0.0        | 0.0              | 0.0066      | 0.0        | 0.0         | 0.0                | 0.5589 | 0.3733          | 0.0           | 0.8152  | 0.0                    | 0.0137    | 0.4013      | 0.3786       | 0.1117          | 0.0    | 0.0       | 0.0                 | 0.0    | 0.0        | 0.0    | 0.0104   | 0.0       | 0.5657           | 0.0       | 0.1786  | 0.7969 | 0.7734       | 0.0710      | 0.2662        | 0.0           | 0.2335     |
| 1.2518        | 2.0   | 670  | 0.1360          | 0.7806            | 0.8283         | 0.8037     | 0.9571           | 0.7286         | 0.6427           | 0.6429    | 0.5102 | 0.6207            | 0.1322            | 0.9476  | 0.1031          | 0.7823    | 0.0303           | 0.0                 | 0.4403              | 0.5190      | 0.0             | 0.0144          | 0.0               | 0.9125  | 0.0            | 0.9908   | 0.7273             | 0.7199       | 0.9762      | 0.0       | 0.2890  | 0.0    | 0.8519  | 0.5472  | 0.8354     | 0.0              | 0.7228      | 0.0        | 0.3513      | 0.0                | 0.8381 | 0.0117          | 0.0           | 0.9740  | 0.0                    | 0.3070    | 0.7378      | 0.8857       | 0.4724          | 0.0    | 0.3978    | 0.4541              | 0.0278 | 0.0        | 0.2254 | 0.7361   | 0.0205    | 0.7132           | 0.0       | 0.9032  | 0.9870 | 0.9540       | 0.7943      | 0.6036        | 0.6184        | 0.6923     |
| 0.1589        | 3.0   | 1005 | 0.0721          | 0.8615            | 0.9008         | 0.8807     | 0.9770           | 0.9164         | 0.9765           | 0.8283    | 0.5200 | 0.8077            | 0.6461            | 0.9790  | 0.6881          | 0.9592    | 0.5217           | 0.6769              | 0.5950              | 0.4094      | 0.5758          | 0.2397          | 0.0               | 0.9672  | 0.0            | 0.9994   | 0.9484             | 0.8170       | 0.9836      | 0.6437    | 0.9492  | 0.0    | 0.8424  | 0.8056  | 0.8999     | 0.0              | 0.7921      | 0.2667     | 0.5761      | 0.0                | 0.9841 | 0.0103          | 0.2147        | 0.9880  | 0.0                    | 0.8051    | 0.8299      | 0.9947       | 0.7793          | 0.5161 | 0.7444    | 0.9894              | 0.7692 | 0.0        | 0.8182 | 0.9939   | 0.5244    | 0.4451           | 0.0       | 0.9762  | 0.9896 | 1.0          | 0.9008      | 0.9349        | 0.9605        | 0.9337     |
| 0.1589        | 4.0   | 1340 | 0.0386          | 0.9175            | 0.9445         | 0.9308     | 0.9876           | 0.9597         | 0.9921           | 0.9041    | 0.9691 | 0.7944            | 0.7662            | 0.9940  | 0.9864          | 0.9801    | 0.7463           | 0.9560              | 0.8562              | 0.7383      | 0.7308          | 0.4286          | 0.0               | 0.9861  | 0.0            | 1.0      | 1.0                | 0.8726       | 0.9916      | 0.8434    | 0.9884  | 0.0382 | 0.8700  | 0.4811  | 0.9517     | 0.0741           | 0.8927      | 0.6732     | 0.7251      | 0.5629             | 1.0    | 0.6341          | 0.3353        | 0.9968  | 0.0                    | 0.9648    | 0.9532      | 0.9947       | 0.9725          | 0.7719 | 0.8683    | 0.9947              | 0.9028 | 0.0        | 0.9302 | 0.9957   | 0.8287    | 0.8698           | 0.1389    | 0.9841  | 0.9974 | 0.9832       | 0.9303      | 0.9639        | 0.9673        | 0.9573     |
| 0.0637        | 5.0   | 1675 | 0.0226          | 0.9402            | 0.9627         | 0.9513     | 0.9936           | 1.0            | 1.0              | 0.9355    | 0.9796 | 0.9813            | 0.8643            | 0.9987  | 0.9640          | 1.0       | 0.9197           | 0.9693              | 0.9368              | 0.7273      | 0.8052          | 0.5455          | 0.1395            | 0.9916  | 0.0615         | 1.0      | 0.9952             | 0.9051       | 0.9933      | 0.9048    | 1.0     | 0.2069 | 0.8700  | 0.5124  | 0.9728     | 0.4444           | 0.9107      | 0.7753     | 0.8147      | 0.9023             | 0.9741 | 0.8521          | 0.5990        | 0.9978  | 0.0                    | 1.0       | 0.9970      | 1.0          | 0.9953          | 0.8713 | 0.8913    | 0.9735              | 0.9583 | 0.0        | 0.9924 | 0.9974   | 0.9041    | 0.9192           | 0.5053    | 0.9801  | 0.9974 | 1.0          | 0.9521      | 1.0           | 0.9934        | 0.975      |
| 0.0333        | 6.0   | 2010 | 0.0136          | 0.9683            | 0.9774         | 0.9728     | 0.9966           | 0.9963         | 1.0              | 0.9454    | 1.0    | 1.0               | 0.9670            | 0.9987  | 1.0             | 1.0       | 0.9481           | 0.9880              | 1.0                 | 0.8475      | 0.8701          | 0.7174          | 0.36              | 0.9944  | 0.4776         | 1.0      | 1.0                | 0.9441       | 0.9982      | 0.9398    | 1.0     | 0.3661 | 0.8519  | 0.7309  | 0.9785     | 0.7108           | 0.9474      | 0.8722     | 0.8909      | 0.9848             | 0.9895 | 1.0             | 0.7           | 0.9994  | 0.5                    | 1.0       | 1.0         | 1.0          | 1.0             | 0.96   | 0.9535    | 0.9947              | 0.9718 | 0.0        | 1.0    | 0.9974   | 0.9810    | 0.9815           | 0.7037    | 0.9880  | 0.9974 | 1.0          | 0.9681      | 1.0           | 1.0           | 0.9938     |
| 0.0333        | 7.0   | 2345 | 0.0106          | 0.9760            | 0.9801         | 0.9780     | 0.9977           | 1.0            | 1.0              | 0.9565    | 1.0    | 1.0               | 0.9753            | 0.9987  | 1.0             | 1.0       | 0.9701           | 0.9939              | 1.0                 | 0.8668      | 0.8662          | 0.7582          | 0.36              | 0.9944  | 0.5970         | 1.0      | 1.0                | 0.9493       | 0.9982      | 0.9524    | 1.0     | 0.5543 | 0.8700  | 0.8863  | 0.9806     | 0.6875           | 0.9424      | 0.8811     | 0.9052      | 0.9848             | 1.0    | 1.0             | 0.7364        | 0.9994  | 0.5                    | 1.0       | 1.0         | 1.0          | 1.0             | 0.9697 | 0.9540    | 0.9947              | 0.9650 | 0.0        | 1.0    | 0.9965   | 0.9810    | 0.9832           | 0.7928    | 0.9880  | 0.9974 | 1.0          | 0.9746      | 1.0           | 1.0           | 0.9969     |


### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1+cu117
- Datasets 2.13.1
- Tokenizers 0.13.3