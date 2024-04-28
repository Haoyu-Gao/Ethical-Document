---
language: en
tags:
- array
- of
- tags
datasets:
- jpwahle/machine-paraphrase-dataset
thumbnail: url to a thumbnail used in social sharing
widget:
- text: Plagiarism is the representation of another author's writing, thoughts, ideas,
    or expressions as one's own work.
---

# Longformer-base for Machine-Paraphrase Detection

If you are using this model in your research work, please cite

```
@InProceedings{10.1007/978-3-030-96957-8_34,
    author="Wahle, Jan Philip and Ruas, Terry and Folt{\'y}nek, Tom{\'a}{\v{s}} and Meuschke, Norman and Gipp, Bela",
    title="Identifying Machine-Paraphrased Plagiarism",
    booktitle="Information for a Better World: Shaping the Global Future",
    year="2022",
    publisher="Springer International Publishing",
    address="Cham",
    pages="393--413",
    abstract="Employing paraphrasing tools to conceal plagiarized text is a severe threat to academic integrity. To enable the detection of machine-paraphrased text, we     evaluate the effectiveness of five pre-trained word embedding models combined with machine learning classifiers and state-of-the-art neural language models. We analyze preprints of research papers, graduation theses, and Wikipedia articles, which we paraphrased using different configurations of the tools SpinBot and SpinnerChief. The best performing technique, Longformer, achieved an average F1 score of 80.99{\%} (F1=99.68{\%} for SpinBot and F1=71.64{\%} for SpinnerChief cases), while human evaluators achieved F1=78.4{\%} for SpinBot and F1=65.6{\%} for SpinnerChief cases. We show that the automated classification alleviates shortcomings of widely-used text-matching systems, such as Turnitin and PlagScan.",
    isbn="978-3-030-96957-8"
}
```

This is the checkpoint for Longformer-base after being trained on the [Machine-Paraphrased Plagiarism Dataset](https://doi.org/10.5281/zenodo.3608000)

Additional information about this model:

* [The longformer-base-4096 model page](https://huggingface.co/allenai/longformer-base-4096)
* [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf)
* [Official implementation by AllenAI](https://github.com/allenai/longformer)

The model can be loaded to perform Plagiarism like so:

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

AutoModelForSequenceClassification("jpelhaw/longformer-base-plagiarism-detection")
AutoTokenizer.from_pretrained("jpelhaw/longformer-base-plagiarism-detection")

input = "Plagiarism is the representation of another author's writing, \
thoughts, ideas, or expressions as one's own work."


example = tokenizer.tokenize(input, add_special_tokens=True)

answer = model(**example)
                                
# "plagiarised"
```