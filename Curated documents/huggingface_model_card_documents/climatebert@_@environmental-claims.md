---
language: en
license: apache-2.0
tags:
- ClimateBERT
- climate
datasets: climatebert/environmental_claims
---

# Model Card for environmental-claims

## Model Description

The environmental-claims model is fine-tuned on the [EnvironmentalClaims](https://huggingface.co/datasets/climatebert/environmental_claims) dataset by using the [climatebert/distilroberta-base-climate-f](https://huggingface.co/climatebert/distilroberta-base-climate-f) model as pre-trained language model. The underlying methodology can be found in our [research paper](https://arxiv.org/abs/2209.00507).

## Climate Performance Model Card

| environmental-claims                                                     |                |
|--------------------------------------------------------------------------|----------------|
| 1. Is the resulting model publicly available?                            | Yes            |
| 2. How much time does the training of the final model take?              | < 5 min        |
| 3. How much time did all experiments take (incl. hyperparameter search)? | 60 hours       |
| 4. What was the power of GPU and CPU?                                    | 0.3 kW         |
| 5. At which geo location were the computations performed?                | Switzerland    |
| 6. What was the energy mix at the geo location?                          | 89 gCO2eq/kWh  |
| 7. How much CO2eq was emitted to train the final model?                  | 2.2 g          |
| 8. How much CO2eq was emitted for all experiments?                       | 1.6 kg         |
| 9. What is the average CO2eq emission for the inference of one sample?   | 0.0067 mg      |
| 10. Which positive environmental impact can be expected from this work?  | This work can help detect and evaluate environmental claims and thus have a positive impact on the environment in the future. |
| 11. Comments                                                             | - |

## Citation Information

```bibtex
@misc{stammbach2022environmentalclaims,
  title = {A Dataset for Detecting Real-World Environmental Claims},
  author = {Stammbach, Dominik and Webersinke, Nicolas and Bingler, Julia Anna and Kraus, Mathias and Leippold, Markus},
  year = {2022},
  doi = {10.48550/ARXIV.2209.00507},
  url = {https://arxiv.org/abs/2209.00507},
  publisher = {arXiv},
}
```

## How to Get Started With the Model

You can use the model with a pipeline for text classification:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

dataset_name = "climatebert/environmental_claims"
model_name = "climatebert/environmental-claims"

# If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
dataset = datasets.load_dataset(dataset_name, split="test")

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
   print(out)
```