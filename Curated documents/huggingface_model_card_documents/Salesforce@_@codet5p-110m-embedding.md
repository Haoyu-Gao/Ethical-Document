---
license: bsd-3-clause
---

# CodeT5+ 110M Embedding Models

## Model description

[CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) is a new family of open code large language models
with an encoder-decoder architecture that can flexibly operate in different modes (i.e. _encoder-only_, _decoder-only_,
and _encoder-decoder_) to support a wide range of code understanding and generation tasks.
It is introduced in the paper:

[CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf)
by [Yue Wang](https://yuewang-cuhk.github.io/)\*, [Hung Le](https://sites.google.com/view/henryle2018/home?pli=1)\*, [Akhilesh Deepak Gotmare](https://akhileshgotmare.github.io/), [Nghi D.Q. Bui](https://bdqnghi.github.io/), [Junnan Li](https://sites.google.com/site/junnanlics), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home) (*
indicates equal contribution).

Compared to the original CodeT5 family (base: `220M`, large: `770M`), CodeT5+ is pretrained with a diverse set of
pretraining tasks including _span denoising_, _causal language modeling_, _contrastive learning_, and _text-code
matching_ to learn rich representations from both unimodal code data and bimodal code-text data.
Additionally, it employs a simple yet effective _compute-efficient pretraining_ method to initialize the model
components with frozen off-the-shelf LLMs such as [CodeGen](https://github.com/salesforce/CodeGen) to efficiently scale
up the model (i.e. `2B`, `6B`, `16B`), and adopts a "shallow encoder and deep decoder" architecture.
Furthermore, it is instruction-tuned to align with natural language instructions (see our InstructCodeT5+ 16B)
following [Code Alpaca](https://github.com/sahil280114/codealpaca).

## How to use

This checkpoint consists of an encoder of CodeT5+ 220M model (pretrained from 2 stages on both unimodal and bimodal) and a projection layer, which can be used to extract code
embeddings of 256 dimension. It can be easily loaded using the `AutoModel` functionality and employs the
same [CodeT5](https://github.com/salesforce/CodeT5) tokenizer.

```python
from transformers import AutoModel, AutoTokenizer

checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
embedding = model(inputs)[0]
print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
# Dimension of the embedding: 256, with norm=1.0
print(embedding)
# tensor([ 0.0185,  0.0229, -0.0315, -0.0307, -0.1421, -0.0575, -0.0275,  0.0501,
#          0.0203,  0.0337, -0.0067, -0.0075, -0.0222, -0.0107, -0.0250, -0.0657,
#          0.1571, -0.0994, -0.0370,  0.0164, -0.0948,  0.0490, -0.0352,  0.0907,
#         -0.0198,  0.0130, -0.0921,  0.0209,  0.0651,  0.0319,  0.0299, -0.0173,
#         -0.0693, -0.0798, -0.0066, -0.0417,  0.1076,  0.0597, -0.0316,  0.0940,
#         -0.0313,  0.0993,  0.0931, -0.0427,  0.0256,  0.0297, -0.0561, -0.0155,
#         -0.0496, -0.0697, -0.1011,  0.1178,  0.0283, -0.0571, -0.0635, -0.0222,
#          0.0710, -0.0617,  0.0423, -0.0057,  0.0620, -0.0262,  0.0441,  0.0425,
#         -0.0413, -0.0245,  0.0043,  0.0185,  0.0060, -0.1727, -0.1152,  0.0655,
#         -0.0235, -0.1465, -0.1359,  0.0022,  0.0177, -0.0176, -0.0361, -0.0750,
#         -0.0464, -0.0846, -0.0088,  0.0136, -0.0221,  0.0591,  0.0876, -0.0903,
#          0.0271, -0.1165, -0.0169, -0.0566,  0.1173, -0.0801,  0.0430,  0.0236,
#          0.0060, -0.0778, -0.0570,  0.0102, -0.0172, -0.0051, -0.0891, -0.0620,
#         -0.0536,  0.0190, -0.0039, -0.0189, -0.0267, -0.0389, -0.0208,  0.0076,
#         -0.0676,  0.0630, -0.0962,  0.0418, -0.0172, -0.0229, -0.0452,  0.0401,
#          0.0270,  0.0677, -0.0111, -0.0089,  0.0175,  0.0703,  0.0714, -0.0068,
#          0.1214, -0.0004,  0.0020,  0.0255,  0.0424, -0.0030,  0.0318,  0.1227,
#          0.0676, -0.0723,  0.0970,  0.0637, -0.0140, -0.0283, -0.0120,  0.0343,
#         -0.0890,  0.0680,  0.0514,  0.0513,  0.0627, -0.0284, -0.0479,  0.0068,
#         -0.0794,  0.0202,  0.0208, -0.0113, -0.0747,  0.0045, -0.0854, -0.0609,
#         -0.0078,  0.1168,  0.0618, -0.0223, -0.0755,  0.0182, -0.0128,  0.1116,
#          0.0240,  0.0342,  0.0119, -0.0235, -0.0150, -0.0228, -0.0568, -0.1528,
#          0.0164, -0.0268,  0.0727, -0.0569,  0.1306,  0.0643, -0.0158, -0.1070,
#         -0.0107, -0.0139, -0.0363,  0.0366, -0.0986, -0.0628, -0.0277,  0.0316,
#          0.0363,  0.0038, -0.1092, -0.0679, -0.1398, -0.0648,  0.1711, -0.0666,
#          0.0563,  0.0581,  0.0226,  0.0347, -0.0672, -0.0229, -0.0565,  0.0623,
#          0.1089, -0.0687, -0.0901, -0.0073,  0.0426,  0.0870, -0.0390, -0.0144,
#         -0.0166,  0.0262, -0.0310,  0.0467, -0.0164, -0.0700, -0.0602, -0.0720,
#         -0.0386,  0.0067, -0.0337, -0.0053,  0.0829,  0.1004,  0.0427,  0.0026,
#         -0.0537,  0.0951,  0.0584, -0.0583, -0.0208,  0.0124,  0.0067,  0.0403,
#          0.0091, -0.0044, -0.0036,  0.0524,  0.1103, -0.1511, -0.0479,  0.1709,
#          0.0772,  0.0721, -0.0332,  0.0866,  0.0799, -0.0581,  0.0713,  0.0218],
#        device='cuda:0', grad_fn=<SelectBackward0>)
```

## Pretraining data

This checkpoint is trained on the stricter permissive subset of the deduplicated version of
the [github-code dataset](https://huggingface.co/datasets/codeparrot/github-code).
The data is preprocessed by reserving only permissively licensed code ("mit" “apache-2”, “bsd-3-clause”, “bsd-2-clause”,
“cc0-1.0”, “unlicense”, “isc”).
Supported languages (9 in total) are as follows:
`c`, `c++`, `c-sharp`,  `go`, `java`, `javascript`,  `php`, `python`, `ruby.`

## Training procedure

This checkpoint is first trained on the unimodal code data at the first-stage pretraining and then on bimodal text-code
pair data using the proposed mixture of pretraining tasks.
Please refer to the paper for more details.

## Evaluation results

We show the zero-shot results of this checkpoint on 6 downstream code retrieval tasks from CodeXGLUE in the following table.
| Ruby  | JavaScript | Go    | Python | Java  | PHP   | Overall |
| ----- | ---------- | ----- | ------ | ----- | ----- | ------- |
| 74.51 | 69.07      | 90.69 | 71.55  | 71.82 | 67.72 | 74.23   |

## BibTeX entry and citation info

```bibtex
@article{wang2023codet5plus,
  title={CodeT5+: Open Code Large Language Models for Code Understanding and Generation},
  author={Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D.Q. and Li, Junnan and Hoi, Steven C. H.},
  journal={arXiv preprint},
  year={2023}
}
```