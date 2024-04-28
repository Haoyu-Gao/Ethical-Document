---
language:
- en
license: mit
library_name: transformers.js
pipeline_tag: feature-extraction
---

_Fork of https://huggingface.co/thenlper/gte-small with ONNX weights to be compatible with Transformers.js. See [JavaScript usage](#javascript)._

---

# gte-small

Gegeral Text Embeddings (GTE) model. 

The GTE models are trained by Alibaba DAMO Academy. They are mainly based on the BERT framework and currently offer three different sizes of models, including [GTE-large](https://huggingface.co/thenlper/gte-large), [GTE-base](https://huggingface.co/thenlper/gte-base), and [GTE-small](https://huggingface.co/thenlper/gte-small). The GTE models are trained on a large-scale corpus of relevance text pairs, covering a wide range of domains and scenarios. This enables the GTE models to be applied to various downstream tasks of text embeddings, including **information retrieval**, **semantic textual similarity**, **text reranking**, etc.

## Metrics

Performance of GTE models were compared with other popular text embedding models on the MTEB benchmark. For more detailed comparison results, please refer to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).



| Model Name | Model Size (GB) | Dimension | Sequence Length | Average (56) | Clustering (11) | Pair Classification (3) | Reranking (4) | Retrieval (15) | STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**gte-large**](https://huggingface.co/thenlper/gte-large) | 0.67 | 1024 | 512 | **63.13** | 46.84 | 85.00 | 59.13 | 52.22 | 83.35 | 31.66 | 73.33 |
| [**gte-base**](https://huggingface.co/thenlper/gte-base) 	| 0.22 | 768 | 512 | **62.39** | 46.2 | 84.57 | 58.61 | 51.14 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1.34 | 1024| 512 | 62.25 | 44.49 | 86.03 | 56.61 | 50.56 | 82.05 | 30.19 | 75.24 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.44 | 768 | 512 | 61.5 | 43.80 | 85.73 | 55.91 | 50.29 | 81.05 | 30.28 | 73.84 |
| [**gte-small**](https://huggingface.co/thenlper/gte-small) | 0.07 | 384 | 512 | **61.36** | 44.89 | 83.54 | 57.7 | 49.46 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | - | 1536 | 8192 | 60.99 | 45.9 | 84.89 | 56.32 | 49.25 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.13 | 384 | 512 | 59.93 | 39.92 | 84.67 | 54.32 | 49.04 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | 9.73 | 768 | 512 | 59.51 | 43.72 | 85.06 | 56.42 | 42.24 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	| 0.44 | 768 | 514 	| 57.78 | 43.69 | 83.04 | 59.36 | 43.81 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	| 28.27 | 4096 | 2048 | 57.59 | 38.93 | 81.9 | 55.65 | 48.22 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	| 0.13 | 384 | 512 	| 56.53 | 41.81 | 82.41 | 58.44 | 42.69 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	| 0.09 | 384 | 512 	| 56.26 | 42.35 | 82.37 | 58.04 | 41.95 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	| 0.44 | 768 | 512 	| 56.00 | 41.1 	| 82.54 | 53.14 | 41.88 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	| 0.22 | 768 | 512 	| 55.27 | 40.21 | 85.18 | 53.09 | 33.63 | 81.14 | 31.39 | 69.81 |


## Usage

This model can be used with both [Python](#python) and [JavaScript](#javascript).

### Python
Use with [Transformers](https://huggingface.co/docs/transformers/index) and [PyTorch](https://pytorch.org/docs/stable/index.html):

```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

tokenizer = AutoTokenizer.from_pretrained("Supabase/gte-small")
model = AutoModel.from_pretrained("Supabase/gte-small")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
```

Use with [sentence-transformers](https://www.sbert.net/):
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ['That is a happy person', 'That is a very happy person']

model = SentenceTransformer('Supabase/gte-small')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
```

### JavaScript
This model can be used with JavaScript via [Transformers.js](https://huggingface.co/docs/transformers.js/index).

Use with [Deno](https://deno.land/manual/introduction) or [Supabase Edge Functions](https://supabase.com/docs/guides/functions):

```ts
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.0'

// Configuration for Deno runtime
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline(
  'feature-extraction',
  'Supabase/gte-small',
);

serve(async (req) => {
  // Extract input string from JSON body
  const { input } = await req.json();

  // Generate the embedding from the user input
  const output = await pipe(input, {
    pooling: 'mean',
    normalize: true,
  });

  // Extract the embedding output
  const embedding = Array.from(output.data);

  // Return the embedding
  return new Response(
    JSON.stringify({ embedding }),
    { headers: { 'Content-Type': 'application/json' } }
  );
});
```

Use within the browser ([JavaScript Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)):

```html
<script type="module">

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.0';

const pipe = await pipeline(
  'feature-extraction',
  'Supabase/gte-small',
);

// Generate the embedding from text
const output = await pipe('Hello world', {
  pooling: 'mean',
  normalize: true,
});

// Extract the embedding output
const embedding = Array.from(output.data);

console.log(embedding);

</script>
```

Use within [Node.js](https://nodejs.org/en/docs) or a web bundler ([Webpack](https://webpack.js.org/concepts/), etc):

```js
import { pipeline } from '@xenova/transformers';

const pipe = await pipeline(
  'feature-extraction',
  'Supabase/gte-small',
);

// Generate the embedding from text
const output = await pipe('Hello world', {
  pooling: 'mean',
  normalize: true,
});

// Extract the embedding output
const embedding = Array.from(output.data);

console.log(embedding);
```

### Limitation

This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.
