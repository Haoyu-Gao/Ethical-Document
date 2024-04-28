---
license: apache-2.0
datasets:
- climatebert/netzero_reduction_data
---
# Model Card for netzero-reduction

## Model Description

Based on [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4599483), this is the fine-tuned ClimateBERT language model with a classification head for detecting sentences that are either related to emission net zero or reduction targets. 
We use the [climatebert/distilroberta-base-climate-f](https://huggingface.co/climatebert/distilroberta-base-climate-f) language model as a starting point and fine-tuned it on our human-annotated dataset.
 
## Citation Information

```bibtex
@article{schimanski2023climatebertnetzero,
      title={ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets}, 
      author={Tobias Schimanski and Julia Bingler and Camilla Hyslop and Mathias Kraus and Markus Leippold},
      year={2023},
      eprint={2310.08096},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## How to Get Started With the Model
You can use the model with a pipeline for text classification:

IMPORTANT REMARK: It is highly recommended to use a prior classification step before applying ClimateBERT-NetZero. Establish a climate context with [climatebert/distilroberta-base-climate-detector](https://huggingface.co/climatebert/distilroberta-base-climate-detector) for paragraphs or [ESGBERT/EnvironmentalBERT-environmental](https://huggingface.co/ESGBERT/EnvironmentalBERT-environmental) for sentences and then label the data with ClimateBERT-NetZero.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm
 
dataset_name = "climatebert/climate_detection"
tokenizer_name = "climatebert/distilroberta-base-climate-f"
model_name = "climatebert/netzero-reduction"
 
# If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
dataset = datasets.load_dataset(dataset_name, split="test")
 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len=512)
 
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
 
# See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
for i, out in enumerate(tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True))):
  print(dataset["text"][i])
  print(out)

### IMPORTANT REMARK: It is highly recommended to use a prior classification step before applying ClimateBERT-NetZero.
### Establish a climate context with "climatebert/distilroberta-base-climate-detector" for paragraphs
### or "ESGBERT/EnvironmentalBERT-environmental" for sentences and then label the data with ClimateBERT-NetZero.
```