---
license: apache-2.0
datasets: google_wellformed_query
inference: false
---
**Intended Use Cases**

*Content Creation*: Validate the well-formedness of written content.

*Educational Platforms*: Helps students check the grammaticality of their sentences.

*Chatbots & Virtual Assistants*: To validate user queries or generate well-formed responses.

**contact: kua613@g.harvard.edu**
  
**Model name**: Query Wellformedness Scoring

**Description** : Evaluate the well-formedness of sentences by checking grammatical correctness and completeness. Sensitive to case and penalizes sentences for incorrect grammar and case.

**Features**:
  - *Wellformedness Score*: Provides a score indicating grammatical correctness and completeness.
  - *Case Sensitivity*: Recognizes and penalizes incorrect casing in sentences.
  - *Broad Applicability*: Can be used on a wide range of sentences.

**Example**:
1. Dogs are mammals.
2. she loves to read books on history.
3. When the rain in Spain.
4. Eating apples are healthy for you.
5. The Eiffel Tower is in Paris.

Among these sentences:
Sentences 1 and 5 are well-formed and have correct grammar and case.
Sentence 2 starts with a lowercase letter.
Sentence 3 is a fragment and is not well-formed.
Sentence 4 has a subject-verb agreement error.


**example_usage:**
*library: HuggingFace transformers*
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("Ashishkr/query_wellformedness_score")
model = AutoModelForSequenceClassification.from_pretrained("Ashishkr/query_wellformedness_score")
sentences = [
    "The quarterly financial report are showing an increase.",  # Incorrect
    "Him has completed the audit for last fiscal year.",  # Incorrect
    "Please to inform the board about the recent developments.",  # Incorrect
    "The team successfully achieved all its targets for the last quarter.",  # Correct
    "Our company is exploring new ventures in the European market."  # Correct
]

features = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    scores = model(**features).logits
print(scores)
```







