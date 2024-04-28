---
{}
---
# <a name="introduction"></a> BERTweet: A pre-trained language model for English Tweets 

BERTweet is the first public large-scale language model pre-trained for English Tweets. BERTweet is trained based on the [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  pre-training procedure. The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related to the **COVID-19** pandemic. The general architecture and experimental results of BERTweet can be found in our [paper](https://aclanthology.org/2020.emnlp-demos.2/):

    @inproceedings{bertweet,
    title     = {{BERTweet: A pre-trained language model for English Tweets}},
    author    = {Dat Quoc Nguyen and Thanh Vu and Anh Tuan Nguyen},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    pages     = {9--14},
    year      = {2020}
    }

**Please CITE** our paper when BERTweet is used to help produce published results or is incorporated into other software.

For further information or requests, please go to [BERTweet's homepage](https://github.com/VinAIResearch/BERTweet)!

### Main results

<p float="left">
<img width="275" alt="postagging" src="https://user-images.githubusercontent.com/2412555/135724590-01d8d435-262d-44fe-a383-cd39324fe190.png" />
<img width="275" alt="ner" src="https://user-images.githubusercontent.com/2412555/135724598-1e3605e7-d8ce-4c5e-be4a-62ae8501fae7.png" />
</p>

<p float="left">
<img width="275" alt="sentiment" src="https://user-images.githubusercontent.com/2412555/135724597-f1981f1e-fe73-4c03-b1ff-0cae0cc5f948.png" />
<img width="275" alt="irony" src="https://user-images.githubusercontent.com/2412555/135724595-15f4f2c8-bbb6-4ee6-82a0-034769dec183.png" />
</p>

