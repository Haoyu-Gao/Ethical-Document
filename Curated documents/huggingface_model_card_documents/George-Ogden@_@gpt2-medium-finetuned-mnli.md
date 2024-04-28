---
language:
- en
license: mit
metrics:
- glue
pipeline_tag: text-classification
---
Evaluate on MNLI:
```python
from transformers import (
    default_data_collator,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset

import functools

from utils import compute_metrics, preprocess_function

model_name = "George-Ogden/gpt2-medium-finetuned-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
trainer = Trainer(
    model=model,
    eval_dataset="mnli",
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
)

raw_datasets = load_dataset(
    "glue",
    "mnli",
).map(functools.partial(preprocess_function, tokenizer), batched=True)

tasks = ["mnli", "mnli-mm"]
eval_datasets = [
    raw_datasets["validation_matched"],
    raw_datasets["validation_mismatched"],
]

for layers in reversed(range(model.num_layers + 1)):
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}

        trainer.log_metrics(metrics)
```