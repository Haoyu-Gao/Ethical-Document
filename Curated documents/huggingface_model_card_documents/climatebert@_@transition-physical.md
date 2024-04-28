---
license: apache-2.0
---
# Model Card for transition-physical

## Model Description

This is the fine-tuned ClimateBERT language model with a classification head for detecting sentences that are either related to transition risks or to physical climate risks.
Using the [climatebert/distilroberta-base-climate-f](https://huggingface.co/climatebert/distilroberta-base-climate-f) language model as starting point, the distilroberta-base-climate-detector model is fine-tuned on our human-annotated dataset.
 
## Citation Information

```bibtex
@article{deng2023war,
  title={War and Policy: Investor Expectations on the Net-Zero Transition},
  author={Deng, Ming and Leippold, Markus and Wagner, Alexander F and Wang, Qian},
  journal={Swiss Finance Institute Research Paper},
  number={22-29},
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
 
dataset_name = "climatebert/climate_detection"
tokenizer_name = “"climatebert/distilroberta-base-climate-detector"
model_name = "climatebert/transition-physical"
 
# If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
dataset = datasets.load_dataset(dataset_name, split="test")
 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len=512)
 
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
 
# See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
   print(out)
```