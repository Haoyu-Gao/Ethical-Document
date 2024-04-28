---
{}
---
This is a RoBERTa-large classifier trained on the CoLA corpus [Warstadt et al., 2019](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00290), 
which contains sentences paired with grammatical acceptability judgments. The model can be used to evaluate fluency of machine-generated English sentences, e.g. for evaluation of text style transfer.

The model was trained in the paper [Krishna et al, 2020. Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700), and its original version is available at [their project page](http://style.cs.umass.edu). We converted this model from Fairseq to Transformers format. All credit goes to the authors of the original paper. 

## Citation
If you found this model useful and refer to it, please cite the original work:
```
@inproceedings{style20,
author={Kalpesh Krishna and John Wieting and Mohit Iyyer},
Booktitle = {Empirical Methods in Natural Language Processing},
Year = "2020",
Title={Reformulating Unsupervised Style Transfer as Paraphrase Generation},
}
```