---
tags:
- protein language model
datasets:
- UniRef50
---

# Encoder only ProtT5-XL-UniRef50, half-precision model

An encoder-only, half-precision version of the [ProtT5-XL-UniRef50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) model. The original model and it's pretraining were introduced in
[this paper](https://doi.org/10.1101/2020.07.12.199554) and first released in
[this repository](https://github.com/agemagician/ProtTrans). This model is trained on uppercase amino acids: it only works with capital letter amino acids. 


## Model description

ProtT5-XL-UniRef50 is based on the `t5-3b` model and was pretrained on a large corpus of protein sequences in a self-supervised fashion.
This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those protein sequences.

One important difference between this T5 model and the original T5 version is the denoising objective.
The original T5-3B model was pretrained using a span denoising objective, while this model was pretrained with a Bart-like MLM denoising objective.
The masking probability is consistent with the original T5 training by randomly masking 15% of the amino acids in the input.

This model only contains the encoder portion of the original ProtT5-XL-UniRef50 model using half precision (float16).
As such, this model can efficiently be used to create protein/ amino acid representations. When used for training downstream networks/ feature extraction, these embeddings produced the same performance (established empirically by comparing on several downstream tasks). 


## Intended uses & limitations

This version of the original ProtT5-XL-UniRef50 is mostly meant for conveniently creating amino-acid or protein embeddings with a low GPU-memory footprint without any measurable performance-decrease in our experiments. This model is fully usable on 8 GB of video RAM.

### How to use

An extensive, interactive example on how to use this model for common tasks can be found [on Google Colab](https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing#scrollTo=ET2v51slC5ui)

Here is how to use this model to extract the features of a given protein sequence in PyTorch:

```python
sequence_examples = ["PRTEINO", "SEQWENCE"]
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

# extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7]) 
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
# do the same for the second ([1,:]) sequence in the batch while taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

print(f"Shape of per-protein embedding of first sequences: {emb_0_per_protein.shape}")
```

**NOTE**: Please make sure to explicitly set the model to `float16` (`T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', torch_dtype=torch.float16)`) otherwise, the generated embeddings will be full precision. 

**NOTE**: Currently (06/2022) half-precision models cannot be used on CPU. If you want to use the encoder only version on CPU, you need to cast it to its full-precision version (`model=model.float()`).

### BibTeX entry and citation info

```bibtex
@article {Elnaggar2020.07.12.199554,
	author = {Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and BHOWMIK, DEBSINDHU and Rost, Burkhard},
	title = {ProtTrans: Towards Cracking the Language of Life{\textquoteright}s Code Through Self-Supervised Deep Learning and High Performance Computing},
	elocation-id = {2020.07.12.199554},
	year = {2020},
	doi = {10.1101/2020.07.12.199554},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Computational biology and bioinformatics provide vast data gold-mines from protein sequences, ideal for Language Models (LMs) taken from Natural Language Processing (NLP). These LMs reach for new prediction frontiers at low inference costs. Here, we trained two auto-regressive language models (Transformer-XL, XLNet) and two auto-encoder models (Bert, Albert) on data from UniRef and BFD containing up to 393 billion amino acids (words) from 2.1 billion protein sequences (22- and 112 times the entire English Wikipedia). The LMs were trained on the Summit supercomputer at Oak Ridge National Laboratory (ORNL), using 936 nodes (total 5616 GPUs) and one TPU Pod (V3-512 or V3-1024). We validated the advantage of up-scaling LMs to larger models supported by bigger data by predicting secondary structure (3-states: Q3=76-84, 8 states: Q8=65-73), sub-cellular localization for 10 cellular compartments (Q10=74) and whether a protein is membrane-bound or water-soluble (Q2=89). Dimensionality reduction revealed that the LM-embeddings from unlabeled data (only protein sequences) captured important biophysical properties governing protein shape. This implied learning some of the grammar of the language of life realized in protein sequences. The successful up-scaling of protein LMs through HPC to larger data sets slightly reduced the gap between models trained on evolutionary information and LMs. Availability ProtTrans: \&lt;a href="https://github.com/agemagician/ProtTrans"\&gt;https://github.com/agemagician/ProtTrans\&lt;/a\&gt;Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554},
	eprint = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554.full.pdf},
	journal = {bioRxiv}
}
```


