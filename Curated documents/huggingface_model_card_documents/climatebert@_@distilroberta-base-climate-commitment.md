---
language:
- en
license: apache-2.0
datasets:
- climatebert/climate_commitments_actions
metrics:
- accuracy
---

# Model Card for distilroberta-base-climate-commitment

## Model Description

This is the fine-tuned ClimateBERT language model with a classification head for classifying climate-related paragraphs into paragraphs being about climate commitments and actions and paragraphs not being about climate commitments and actions.

Using the [climatebert/distilroberta-base-climate-f](https://huggingface.co/climatebert/distilroberta-base-climate-f) language model as starting point, the distilroberta-base-climate-commitment model is fine-tuned on our [climatebert/climate_commitments_actions](https://huggingface.co/climatebert/climate_commitments_actions) dataset.

*Note: This model is trained on paragraphs. It may not perform well on sentences.*

## Citation Information

```bibtex
@techreport{bingler2023cheaptalk,
    title={How Cheap Talk in Climate Disclosures Relates to Climate Initiatives, Corporate Emissions, and Reputation Risk},
    author={Bingler, Julia and Kraus, Mathias and Leippold, Markus and Webersinke, Nicolas},
    type={Working paper},
    institution={Available at SSRN 3998435},
    year={2023}
}
```

## How to Get Started With the Model

You can use the model with a pipeline for text classification:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

dataset_name = "climatebert/climate_commitments_actions"
model_name = "climatebert/distilroberta-base-climate-commitment"

# If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
dataset = datasets.load_dataset(dataset_name, split="test")

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
   print(out)
```