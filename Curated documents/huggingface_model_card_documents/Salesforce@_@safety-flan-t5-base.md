---
{}
---

# Model Details

## Model Description

- **Model type:** Language model
- **License:** CC BY-NC 
- **Related Models:** [All FLAN-T5 Checkpoints](https://huggingface.co/models?search=flan-t5)
- **Resources for more information:**
  - [GitHub Repo](https://github.com/salesforce/AuditNLG)

# Usage

Find below some example scripts on how to use the model in `transformers`:

## Using the Pytorch model

### Running the model

<details>
<summary> Click to expand </summary>

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

config = AutoConfig.from_pretrained("Salesforce/safety-flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/safety-flan-t5-base") 
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/safety-flan-t5-base", config=config)

prefix = "Is the <Text> field safe or unsafe?"
input_context = "Can you teach me this?"
input_text = "You are so stupid"
input_ids = tokenizer(prefix + " <Text> " + input_text + " <Context> " + input_context, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>