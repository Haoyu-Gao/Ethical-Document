<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Anomalous Behavior Profiling (ABP)

# Model Overview

## Description:

* This model is an example of a binary XGBoost classifier to differentiate between anomalous GPU behavior, such as crypto mining / GPU malware, and non-anomalous GPU-based workflows (e.g., ML/DL training). This model is for demonstration purposes and not for production usage. <br>

## References(s):

* Chen, Guestrin (2016) XGBoost. A scalable tree boosting system. https://arxiv.org/abs/1603.02754  <br> 

## Model Architecture: 

**Architecture Type:** 

* Gradient boosting <br>

**Network Architecture:** 

* XGBOOST <br>

## Input: (Enter "None" As Needed)

**Input Format:** 

* nvidia-smi output <br>

**Input Parameters:** 

* GPU statistics that are included in the nvidia-smi output <br>

**Other Properties Related to Output:** N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:** 

* Binary Results <br>

**Output Parameters:** 

* N/A <br>

**Other Properties Related to Output:** 

* N/A <br> 

## Software Integration:

**Runtime(s):** 

* Morpheus  <br>

**Supported Hardware Platform(s):** <br>

* Ampere/Turing <br>

**Supported Operating System(s):** <br>

* Linux <br>

## Model Version(s): 

* v1  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/abp-sample-nvsmi-training-data.json  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Sample dataset consists of over 1000 nvidia-smi outputs <br>

## Evaluation Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/abp-validation-data.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Sample dataset consists of over 1000 nvidia-smi outputs <br>

## Inference:

**Engine:** 

* Triton <br>

**Test Hardware:** <br>

* DGX (V100) <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below.  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Subcards

## Model Card ++ Bias Subcard

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.

* Not Applicable

### Describe measures taken to mitigate against unwanted bias.

* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 

* The model is primarily designed for testing purposes and serves as a small model specifically used to evaluate and validate the ABP pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.

* The model is primarily designed for testing purposes. This model is intended to be an example for developers that want to test Morpheus ABP pipeline.

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to test the functionality of the ABP models for detecting crypto mining.

### Describe the model output. 

* This model output can be used as a binary result, Crypto mining or legitimate GPU usage. 

### List the steps explaining how this model works.  

* nvidia-smi features are used as the input and the model predicts a label for each row 

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:

* Not Applicable

### List the technical limitations of the model. 

* For different GPU workloads different models need to be trained.

### Has this been verified to have met prescribed NVIDIA standards?

* Yes
  
### What performance metrics were used to affirm the model's performance?

* Accuracy

### What are the potential known risks to users and stakeholders?

* N/A

### Link the relevant end user license agreement 

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.

* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/abp-sample-nvsmi-training-data.json

### Is the model used in an application with physical safety impact?

* No

### Describe life-critical impact (if present).

* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?

* No

### Name applications for the model.

* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.

* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

### Name target quality Key Performance Indicators (KPIs) for which this has been tested. 

* N/A

### Is the model and dataset compliant with National Classification Management Society (NCMS)?

* No

### Are there explicit model and dataset restrictions?

* No

### Are there access restrictions to systems, model, and data?

* No

### Is there a digital signature?

* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?

* None

### Was consent obtained for any PII used?

* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)

* N/A
  

### How often is dataset reviewed?

* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?

* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 

* N/A

### Is data in dataset traceable?

* N/A

### Are we able to identify and trace source of dataset?

* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?

* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?

* N/A


<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Model Overview

## Description:
This use case is currently implemented to detect changes in users' behavior that indicate a change from a human to a machine or a machine to a human. The model architecture consists of an Autoencoder, where the reconstruction loss of new log data is used as an anomaly score.

## References(s):
- https://github.com/AlliedToasters/dfencoder/blob/master/dfencoder/autoencoder.py
- Rasheed Peng Alhajj Rokne Jon: Fourier Transform Based Spatial Outlier Mining 2009 - https://link.springer.com/chapter/10.1007/978-3-642-04394-9_39

## Model Architecture:
The model architecture consists of an Autoencoder, where the reconstruction loss of new log data is used as an anomaly score.

**Architecture Type:**
* Autoencoder

**Network Architecture:**
* The network architecture of the model includes a 2-layer encoder with dimensions [512, 500] and a 1-layer decoder with dimensions [512]

## Input:
**Input Format:**
* AWS CloudTrail logs in json format

**Input Parameters:**
* None

**Other Properties Related to Output:**
* Not Applicable (N/A)

## Output:
**Output Format:**
* Anomaly score and the reconstruction loss for each feature in a pandas dataframe

**Output Parameters:**
* None

**Other Properties Related to Output:**
* Not Applicable

## Software Integration:
**Runtime(s):**
* Morpheus

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux<br>

## Model Version(s):
* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/dfp-models/hammah-role-g-20211017-dill.pkl
* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/dfp-models/hammah-user123-20211017-dill.pkl

# Training & Evaluation:

## Training Dataset:

**Link:**
* https://github.com/nv-morpheus/Morpheus/tree/branch-24.03/models/datasets/training-data/cloudtrail

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

The training dataset consists of AWS CloudTrail logs. It contains logs from two entities, providing information about their activities within the AWS environment.
* [hammah-role-g-training-part1.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/cloudtrail/hammah-role-g-training-part1.json): 700 records <br>
* [hammah-role-g-training-part2.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/cloudtrail/hammah-role-g-training-part2.json): 1187 records <br>
* [hammah-user123-training-part2.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/cloudtrail/hammah-user123-training-part2.json): 1000 records <br>
* [hammah-user123-training-part3.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/cloudtrail/hammah-user123-training-part3.json): 1000 records <br>
* [hammah-user123-training-part4.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/cloudtrail/hammah-user123-training-part4.json): 387 records <br>

## Evaluation Dataset:
**Link:**
* https://github.com/nv-morpheus/Morpheus/tree/branch-24.03/models/datasets/validation-data/cloudtrail <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

The evaluation dataset consists of AWS CloudTrail logs. It contains logs from two entities, providing information about their activities within the AWS environment.
* [hammah-role-g-validation.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/cloudtrail/hammah-role-g-validation.json): 314 records
* [hammah-user123-validation-part1.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/cloudtrail/hammah-user123-validation-part1.json): 300 records
* [hammah-user123-validation-part2.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/cloudtrail/hammah-user123-validation-part2.json): 300 records
* [hammah-user123-validation-part3.json](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/cloudtrail/hammah-user123-validation-part3.json): 247 records

## Inference:
**Engine:**
* PyTorch

**Test Hardware:**
* Other

# Subcards

## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?
* Not Applicable

### What is the racial/ethnicity balance of the model validation data?
* Not Applicable

### What is the age balance of the model validation data?
* Not Applicable

### What is the language balance of the model validation data?
* English (cloudtrail logs): 100%

### What is the geographic origin language balance of the model validation data?
* Not Applicable

### What is the educational background balance of the model validation data?
* Not Applicable

### What is the accent balance of the model validation data?
* Not Applicable

### What is the face/key point balance of the model validation data?
* Not Applicable

### What is the skin/tone balance of the model validation data?
* Not Applicable

### What is the religion balance of the model validation data?
* Not Applicable

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.
* Not Applicable

### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model.
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate the DFP pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.
* This model is designed for developers seeking to test the DFP pipeline with a small pretrained model trained on a synthetic dataset.

### Name who is intended to benefit from this model.
* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the DFP pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world cloudtrail logs analysis.

### Describe the model output.
* The model calculates an anomaly score for each input based on the reconstruction loss obtained from the trained Autoencoder. This score represents the level of anomaly detected in the input data. Higher scores indicate a higher likelihood of anomalous behavior.
* The model provides the reconstruction loss of each feature to facilitate further testing and debugging of the pipeline.

### List the steps explaining how this model works.
* The model works by training on baseline behaviors and subsequently detecting deviations from the established baseline, triggering alerts accordingly.
* [Training notebook](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/training-tuning-scripts/dfp-models/hammah-20211017.ipynb)

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* The model expects cloudtrail logs with specific features that match the training dataset. Data lacking the required features or requiring a different feature set may not be compatible with the model.

### What performance metrics were used to affirm the model's performance?
* The model's performance was evaluated based on its ability to correctly identify anomalous behavior in the synthetic dataset during testing.

### What are the potential known risks to users and stakeholders?
* None

### Link the relevant end user license agreement
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* https://github.com/nv-morpheus/Morpheus/tree/branch-24.03/models/datasets/training-data/cloudtrail

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.
* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.
* None

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* No

### Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No

### Is there a digital signature?
* No


## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* The synthetic data used in this model is generated using the [faker](https://github.com/joke2k/faker/blob/master/LICENSE.txt)  python package. The user agent field is generated by faker, which pulls items from its own dataset of fictitious values (located in the linked repo). Similarly, the event source field is randomly chosen from a list of event names provided in the AWS documentation. There are no privacy concerns or PII involved in this synthetic data generation process.

### Protected classes used to create this model? (The following were used in model the model's training:)
* Not applicable

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* No (as the dataset is fully synthetic)

### If PII collected for the development of this AI model, was it minimized to only what was required?
* Not Applicable (no PII collected)

### Is data in dataset traceable?
* No

### Are we able to identify and trace source of dataset?
* Yes ([fully synthetic dataset](https://github.com/nv-morpheus/Morpheus/tree/branch-24.03/models/datasets/training-data/cloudtrail))

### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable (as the dataset is fully synthetic)

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable (as the dataset is fully synthetic)


<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Model Overview

### Description:
* This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of `GraphSAGE` along `XGBoost` is used to identify frauds in the transaction networks.  This model is for demonstration purposes and not for production usage. <br>

## References(s):
1. https://stellargraph.readthedocs.io/en/stable/hinsage.html?highlight=hinsage
2. https://github.com/rapidsai/clx/blob/branch-22.12/examples/forest_inference/xgboost_training.ipynb
3. Rafaël Van Belle, Charles Van Damme, Hendrik Tytgat, Jochen De Weerdt,Inductive Graph Representation Learning for fraud detection (https://www.sciencedirect.com/science/article/abs/pii/S0957417421017449)<br> 

## Model Architecture:
It uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.<br>
**Architecture Type:** 
* Graph Neural Network and Binary classification <br>

**Network Architecture:** 
* GraphSAGE and XGBoost <br>

## Input
Transaction data with nodes including transaction, client, and merchant.<br>
**Input Parameters:**  
* None <br>

**Input Format:** 
* CSV format<br>

**Other Properties Related to Output:** 
* None<br>

## Output
An anomalous score of transactions indicates a probability score of being a fraud.<br>
**Output Parameters:**  
* None <br>

**Output Format:** 
* CSV<br>

**Other Properties Related to Output:** 
* None <br> 

## Software Integration:
**Runtime(s):** 
* Morpheus  <br>

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>
  
## Model Version(s): 
* 1.0 <br>

### How To Use This Model
This model is an example of a fraud detection pipeline using a graph neural network and gradient boosting trees. This can be further retrained or fine-tuned to be used for similar types of transaction networks with similar graph structures.

# Training & Evaluation: 

## Training Dataset:

**Link:**
* [fraud-detection-training-data.csv](models/dataset/fraud-detection-training-data.csv)  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* A training data consists of raw 753 synthetic labeled credit card transaction data with data augmentation in a total of 12053 labeled transaction data. <br>

## Evaluation Dataset:
**Link:**  
* [fraud-detection-validation-data.csv](models/dataset/fraud-detection-validation-data.csv)  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* Data consists of raw 265 labeled credit card transaction synthetically created<br>

## Inference:
**Engine:** 
* Triton <br>

**Test Hardware:** <br>
* DGX (V100) <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below.  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Subcards
## Model Card ++ Bias Subcard

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.
* Not Applicable

### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate the GNN FSI pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.
* This model is designed for developers seeking to test the GNN fraud detection pipeline with a small pretrained model on a synthetic dataset.

### Name who is intended to benefit from this model. 
* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the GNN fraud detection pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world transactions. 

### Describe the model output.
* This model outputs fraud probability score b/n (0 & 1). 

### List the steps explaining how this model works. (e.g., )  
* The model uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.<br>

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* This model version requires a transactional data schema with entities (user, merchant, transaction) as requirement for the model.

### Has this been verified to have met prescribed NVIDIA standards?

* Yes

### What performance metrics were used to affirm the model's performance?
* Area under ROC curve and Accuracy

### What are the potential known risks to users and stakeholders? 
* None

### Link the relevant end user license agreement 
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/LICENSE)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* [training dataset](models/datasets/training-data/fraud-detection-training-data.csv)

### Is the model used in an application with physical safety impact?
* No

### Describe life-critical impact (if present).
* Not Applicable

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* Used for testing fraud detection application in Morpheus pipeline, under the defined dataset schema description.

### Name use case restrictions for the model.
* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* Not Applicable

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Not Applicable

### Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No

### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard

### Generatable or reverse engineerable personally-identifiable information (PII)?
* None

### Protected classes used to create this model? (The following were used in model the model's training:)
* Not applicable

### Was consent obtained for any PII used?
* Not Applicable (Data is extracted from synthetically created credit card transaction,refer[3] for the source of data creation)

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data
* Yes

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* Not applicable

### Is data in dataset traceable?
* No

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable


<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Phishing Detection

# Model Overview

## Description:
* Phishing detection is a binary classifier differentiating between phishing/spam and benign emails and SMS messages.  This model is for demonstration purposes and not for production usage. <br>

## References(s):
* https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection <br>
* Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding https://arxiv.org/abs/1810.04805 <br> 

## Model Architecture: 

**Architecture Type:** 

* Transformers <br>

**Network Architecture:** 

* BERT <br>

## Input: (Enter "None" As Needed)

**Input Format:** 

* Evaluation script downloads the smsspamcollection.zip and extract tabular information into a dataframe <br>

**Input Parameters:** 

* SMS/emails <br>

**Other Properties Related to Output:** 

* N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:** 

* Binary Results, Fraudulent or Benign <br>

**Output Parameters:** 

* N/A <br>

**Other Properties Related to Output:** 

* N/A <br> 


## Software Integration:

**Runtime(s):** 

* Morpheus  <br>

**Supported Hardware Platform(s):** <br>

* Ampere/Turing <br>

**Supported Operating System(s):** <br>

* Linux <br>

## Model Version(s): 

* v1  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:**  

* http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Dataset consists of SMSs <br>

## Evaluation Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/phishing-email-validation-data.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Dataset consists of SMSs <br>

## Inference:

**Engine:** 

* Triton <br>

**Test Hardware:** <br>

* DGX (V100) <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below.  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Subcards

## Model Card ++ Bias Subcard

### What is the language balance of the model validation data?

* English

### What is the geographic origin language balance of the model validation data?

* UK

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.

* Not Applicable

### Describe measures taken to mitigate against unwanted bias.

* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* The model is primarily designed for testing purposes and serves as a small pre-trained model specifically used to evaluate and validate the phishing detection pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.

* This model is designed for developers seeking to test the phishing detection pipeline with a small pre-trained model.

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the phishing pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world phishing messages. 

### Describe the model output. 
* This model output can be used as a binary result, Phishing/Spam or Benign 

### List the steps explaining how this model works.  
* A BERT model gets fine-tuned with the dataset and in the inference it predicts one of the binary classes. Phishing/Spam or Benign.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 
* For different email/SMS types and content, different models need to be trained.

### Has this been verified to have met prescribed NVIDIA standards?

* Yes

### What performance metrics were used to affirm the model's performance?
* F1

### What are the potential known risks to users and stakeholders?
* N/A

### Link the relevant end user license agreement 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.
* http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

### Is the model used in an application with physical safety impact?
* No

### Describe life-critical impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

* The primary application for this model is testing the Morpheus phishing detection pipeline

### Name use case restrictions for the model.
* This pretrained model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* N/A

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* No

### Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No

### Is there a digital signature?

* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* None

### Protected classes used to create this model? (The following were used in model the model's training:)
* N/A

### Was consent obtained for any PII used?
* N/A

### How often is dataset reviewed?
* Unknown

### Is a mechanism in place to honor data subject right of access or deletion of personal data?

* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* N/A

### Are we able to identify and trace source of dataset?
* N/A

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A


<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Root Cause Analysis

# Model Overview

## Description:

* Root cause analysis is a binary classifier differentiating between ordinary logs and errors/problems/root causes in the log files. <br>

## References(s):

* Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding https://arxiv.org/abs/1810.04805 <br> 

## Model Architecture: 

**Architecture Type:** 

* Transformers <br>

**Network Architecture:** 

* BERT <br>

## Input: (Enter "None" As Needed)

**Input Format:** 

* CSV <br>

**Input Parameters:** 

* kern.log file contents <br>

**Other Properties Related to Output:** 

* N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:** 

* Binary Results, Root Cause or Ordinary <br>

**Output Parameters:** 

* N/A <br>

**Other Properties Related to Output:** 

* N/A <br> 

## Software Integration:

**Runtime(s):** 

* Morpheus  <br>

**Supported Hardware Platform(s):** <br>

* Ampere/Turing <br>

**Supported Operating System(s):** <br>

* Linux <br>

## Model Version(s): 
* v1  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/root-cause-training-data.csv <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* kern.log files from DGX machines <br>

## Evaluation Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/validation-data/root-cause-validation-data-input.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* kern.log files from DGX machines <br>

## Inference:

**Engine:** 

* Triton <br>

**Test Hardware:** <br>

* Other  <br>

# Subcards

## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?  
* Not Applicable

### What is the racial/ethnicity balance of the model validation data?
* Not Applicable

### What is the age balance of the model validation data?
* Not Applicable

### What is the language balance of the model validation data?
* Not Applicable

### What is the geographic origin language balance of the model validation data?
* Not Applicable

### What is the educational background balance of the model validation data?
* Not Applicable

### What is the accent balance of the model validation data?
* Not Applicable

### What is the face/key point balance of the model validation data? 
* Not Applicable

### What is the skin/tone balance of the model validation data?
* Not Applicable

### What is the religion balance of the model validation data?
* Not Applicable

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.
* Not Applicable

### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* The model is primarily designed for testing purposes and serves as a small pre-trained model specifically used to evaluate and validate the Root Cause Analysis pipeline. This model is an example of customized transformer-based root cause analysis. It can be used for pipeline testing purposes. It needs to be re-trained for specific root cause analysis or predictive maintenance needs with the fine-tuning scripts in the repo. The hyperparameters can be optimised to adjust to get the best results with another dataset. The aim is to get the model to predict some false positives that could be previously unknown error types. Users can use this root cause analysis approach with other log types too. If they have known failures in their logs, they can use them to train along with ordinary logs and can detect other root causes they weren't aware of before.

### Fill in the blank for the model technique.

* This model is designed for developers seeking to test the root cause analysis pipeline with a small pre-trained model trained on a very small `kern.log` file from a DGX.

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to test the functionality of the DFP pipeline using synthetic datasets

### Describe the model output. 
* This model output can be used as a binary result, Root cause or Ordinary 

### List the steps explaining how this model works.  
* A BERT model gets fine-tuned with the kern.log dataset and in the inference it predicts one of the binary classes. Root cause or Ordinary.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 
* For different log types and content, different models need to be trained.

### What performance metrics were used to affirm the model's performance?
* F1

### What are the potential known risks to users and stakeholders?
* N/A

### Link the relevant end user license agreement 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.
* https://github.com/nv-morpheus/Morpheus/blob/branch-24.03/models/datasets/training-data/root-cause-training-data.csv

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.
* Different models need to be trained depending on the log types.

### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* N/A

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* No

### Are there explicit model and dataset restrictions?
* It is for pipeline testing purposes.

### Are there access restrictions to systems, model, and data?
* No

### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* N/A

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* Original raw logs are not saved. The small sample in the repo is saved for testing the pipeline. 

### Are we able to identify and trace source of dataset?
* N/A

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A