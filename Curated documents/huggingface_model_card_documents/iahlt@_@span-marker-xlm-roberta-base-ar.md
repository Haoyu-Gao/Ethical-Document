---
language:
- ar
license: other
library_name: span-marker
tags:
- span-marker
- token-classification
- ner
- named-entity-recognition
- generated_from_span_marker_trainer
datasets:
- wikiann
metrics:
- precision
- recall
- f1
widget:
- text: جامعة بيزا (إيطاليا).
- text: تعلم في جامعة أوكسفورد، جامعة برنستون، جامعة كولومبيا.
- text: موطنها بلاد الشام تركيا.
- text: عادل إمام - نور الشريف
- text: فوكسي و بورتشا ضد مونكي دي لوفي و نامي
pipeline_tag: token-classification
base_model: xlm-roberta-base
model-index:
- name: SpanMarker with xlm-roberta-base on wikiann
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      name: Unknown
      type: wikiann
      split: eval
    metrics:
    - type: f1
      value: 0.8965362325351544
      name: F1
    - type: precision
      value: 0.9077510917030568
      name: Precision
    - type: recall
      value: 0.8855951007366646
      name: Recall
---

# SpanMarker(Arabic) with xlm-roberta-base on wikiann


This is a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) model trained on the [wikiann](https://huggingface.co/datasets/wikiann) dataset that can be used for Named Entity Recognition. This SpanMarker model uses [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) as the underlying encoder.

## Model Details

### Model Description
- **Model Type:** SpanMarker
- **Encoder:** [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- **Maximum Sequence Length:** 512 tokens
- **Maximum Entity Length:** 30 words
- **Training Dataset:** [wikiann](https://huggingface.co/datasets/wikiann)
- **Languages:** ar
- **License:** other

### Model Sources

- **Repository:** [SpanMarker on GitHub](https://github.com/tomaarsen/SpanMarkerNER)
- **Thesis:** [SpanMarker For Named Entity Recognition](https://raw.githubusercontent.com/tomaarsen/SpanMarkerNER/main/thesis.pdf)

### Model Labels
| Label | Examples                                                               |
|:------|:-----------------------------------------------------------------------|
| LOC   | "شور بلاغ ( مقاطعة غرمي )", "دهنو ( تایباد )", "أقاليم ما وراء البحار" |
| ORG   | "الحزب الاشتراكي", "نادي باسوش دي فيريرا", "دايو ( شركة )"             |
| PER   | "فرنسوا ميتيران،", "ديفيد نالبانديان", "حكم ( كرة قدم )"               |

## Uses

### Direct Use for Inference

```python
from span_marker import SpanMarkerModel

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained("span_marker_model_id")
# Run inference
entities = model.predict("موطنها بلاد الشام تركيا.")
```

### Downstream Use
You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

```python
from span_marker import SpanMarkerModel, Trainer

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained("span_marker_model_id")

# Specify a Dataset with "tokens" and "ner_tag" columns
dataset = load_dataset("conll2003") # For example CoNLL2003

# Initialize a Trainer using the pretrained model & dataset
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
trainer.save_model("span_marker_model_id-finetuned")
```
</details>

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set          | Min | Median | Max |
|:----------------------|:----|:-------|:----|
| Sentence length       | 3   | 6.4592 | 63  |
| Entities per sentence | 1   | 1.1251 | 13  |

### Training Hyperparameters
- learning_rate: 1e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 10

### Training Results
| Epoch  | Step  | Validation Loss | Validation Precision | Validation Recall | Validation F1 | Validation Accuracy |
|:------:|:-----:|:---------------:|:--------------------:|:-----------------:|:-------------:|:-------------------:|
| 0.1989 | 500   | 0.1735          | 0.2667               | 0.0011            | 0.0021        | 0.4103              |
| 0.3979 | 1000  | 0.0808          | 0.7283               | 0.5314            | 0.6145        | 0.7716              |
| 0.5968 | 1500  | 0.0595          | 0.7876               | 0.6872            | 0.7340        | 0.8546              |
| 0.7957 | 2000  | 0.0532          | 0.8148               | 0.7600            | 0.7865        | 0.8823              |
| 0.9946 | 2500  | 0.0478          | 0.8485               | 0.8028            | 0.8250        | 0.9085              |
| 1.1936 | 3000  | 0.0419          | 0.8586               | 0.8084            | 0.8327        | 0.9101              |
| 1.3925 | 3500  | 0.0390          | 0.8628               | 0.8367            | 0.8495        | 0.9237              |
| 1.5914 | 4000  | 0.0456          | 0.8559               | 0.8299            | 0.8427        | 0.9231              |
| 1.7903 | 4500  | 0.0375          | 0.8682               | 0.8469            | 0.8574        | 0.9282              |
| 1.9893 | 5000  | 0.0323          | 0.8821               | 0.8635            | 0.8727        | 0.9348              |
| 2.1882 | 5500  | 0.0346          | 0.8781               | 0.8632            | 0.8706        | 0.9346              |
| 2.3871 | 6000  | 0.0318          | 0.8953               | 0.8523            | 0.8733        | 0.9345              |
| 2.5860 | 6500  | 0.0311          | 0.8861               | 0.8691            | 0.8775        | 0.9373              |
| 2.7850 | 7000  | 0.0323          | 0.89                 | 0.8689            | 0.8793        | 0.9383              |
| 2.9839 | 7500  | 0.0310          | 0.8892               | 0.8780            | 0.8836        | 0.9419              |
| 3.1828 | 8000  | 0.0320          | 0.8817               | 0.8762            | 0.8790        | 0.9397              |
| 3.3817 | 8500  | 0.0291          | 0.8981               | 0.8778            | 0.8878        | 0.9438              |
| 3.5807 | 9000  | 0.0336          | 0.8972               | 0.8792            | 0.8881        | 0.9450              |
| 3.7796 | 9500  | 0.0323          | 0.8927               | 0.8757            | 0.8841        | 0.9424              |
| 3.9785 | 10000 | 0.0315          | 0.9028               | 0.8748            | 0.8886        | 0.9436              |
| 4.1774 | 10500 | 0.0330          | 0.8984               | 0.8855            | 0.8919        | 0.9458              |
| 4.3764 | 11000 | 0.0315          | 0.9023               | 0.8844            | 0.8933        | 0.9469              |
| 4.5753 | 11500 | 0.0305          | 0.9029               | 0.8886            | 0.8957        | 0.9486              |
| 4.6171 | 11605 | 0.0323          | 0.9078               | 0.8856            | 0.8965        | 0.9487              |

### Framework Versions
- Python: 3.10.12
- SpanMarker: 1.4.0
- Transformers: 4.34.1
- PyTorch: 2.1.0+cu118
- Datasets: 2.14.6
- Tokenizers: 0.14.1

## Citation


If you use this model, please cite:
```
@InProceedings{iahlt2023WikiANNArabicNER,
        author =      "iahlt",
        title =       "Arabic NER on WikiANN",
        year =        "2023",
        publisher =   "",
        location =    "",
      }
```


<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->