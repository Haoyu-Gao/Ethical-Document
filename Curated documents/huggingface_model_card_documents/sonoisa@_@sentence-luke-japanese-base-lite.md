---
language: ja
license: apache-2.0
tags:
- sentence-transformers
- sentence-bert
- sentence-luke
- feature-extraction
- sentence-similarity
---

This is a Japanese sentence-LUKE model.

日本語用Sentence-LUKEモデルです。

[日本語Sentence-BERTモデル](https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2)と同一のデータセットと設定で学習しました。  
手元の非公開データセットでは、[日本語Sentence-BERTモデル](https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2)と比べて定量的な精度が同等〜0.5pt程度高く、定性的な精度は本モデルの方が高い結果でした。

事前学習済みモデルとして[studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)を利用させていただきました。  

推論の実行にはSentencePieceが必要です（pip install sentencepiece）。


# 使い方

```python
from transformers import MLukeTokenizer, LukeModel
import torch


class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
model = SentenceLukeJapanese(MODEL_NAME)

sentences = ["暴走したAI", "暴走した人工知能"]
sentence_embeddings = model.encode(sentences, batch_size=8)

print("Sentence embeddings:", sentence_embeddings)
```

