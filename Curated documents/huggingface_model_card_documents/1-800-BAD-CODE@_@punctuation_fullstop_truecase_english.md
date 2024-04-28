---
language:
- en
license: apache-2.0
library_name: generic
tags:
- text2text-generation
- punctuation
- true-casing
- sentence-boundary-detection
- nlp
widget:
- text: hey man how's it going i haven't seen you in a while let's meet at 6 pm for
    drinks
- text: hello user this is an example input this text should be split into several
    sentences including a final interrogative did it work
inference: true
---

# Model Overview
This model accepts as input lower-cased, unpunctuated English text and performs in one pass punctuation restoration, true-casing (capitalization), and sentence boundary detection (segmentation).

In contast to many similar models, this model can predict punctuated acronyms (e.g., "U.S.") via a special "acronym" class, as well as arbitarily-capitalized words (NATO, McDonald's, etc.) via multi-label true-casing predictions.

**Widget note**: The text generation widget doesn't seem to respect line breaks. 
Instead, the pipeline inserts a new line token `\n` in the text where the model has predicted sentence boundaries (line breaks).

# Usage
The easy way to use this model is to install [punctuators](https://github.com/1-800-BAD-CODE/punctuators):

```bash
pip install punctuators
```

If this package is broken, please let me know in the community tab (I update it for each model and break it a lot!).

Let's punctuate my weekend recap, as well as few interesting sentences with acronyms and abbreviations that I made up or found on Wikipedia:

<details open>

  <summary>Example Usage</summary>

```
from typing import List

from punctuators.models import PunctCapSegModelONNX

# Instantiate this model
# This will download the ONNX and SPE models. To clean up, delete this model from your HF cache directory.
m = PunctCapSegModelONNX.from_pretrained("pcs_en")

# Define some input texts to punctuate
input_texts: List[str] = [
    # Literally my weekend
    "i woke up at 6 am and took the dog for a hike in the metacomet mountains we like to take morning adventures on the weekends",
    "despite being mid march it snowed overnight and into the morning here in connecticut it was snowier up in the mountains than in the farmington valley where i live",
    "when i got home i trained this model on the lambda cloud on an a100 gpu with about 10 million lines of text the total budget was less than 5 dollars",
    # Real acronyms in sentences that I made up
    "george hw bush was the president of the us for 8 years",
    "i saw mr smith at the store he was shopping for a new lawn mower i suggested he get one of those new battery operated ones they're so much quieter",
    # See how the model performs on made-up acronyms 
    "i went to the fgw store and bought a new tg optical scope",
    # First few sentences from today's featured article summary on wikipedia
    "it's that man again itma was a radio comedy programme that was broadcast by the bbc for twelve series from 1939 to 1949 featuring tommy handley in the central role itma was a character driven comedy whose satirical targets included officialdom and the proliferation of minor wartime regulations parts of the scripts were rewritten in the hours before the broadcast to ensure topicality"
]
results: List[List[str]] = m.infer(input_texts)
for input_text, output_texts in zip(input_texts, results):
    print(f"Input: {input_text}")
    print(f"Outputs:")
    for text in output_texts:
        print(f"\t{text}")
    print()

```

Exact output may vary based on the model version; here is the current output: 

</details>

<details open>

  <summary>Expected Output</summary>

```text
In: i woke up at 6 am and took the dog for a hike in the metacomet mountains we like to take morning adventures on the weekends
	Out: I woke up at 6 a.m. and took the dog for a hike in the Metacomet Mountains.
	Out: We like to take morning adventures on the weekends.

In: despite being mid march it snowed overnight and into the morning here in connecticut it was snowier up in the mountains than in the farmington valley where i live
	Out: Despite being mid March, it snowed overnight and into the morning.
	Out: Here in Connecticut, it was snowier up in the mountains than in the Farmington Valley where I live.

In: when i got home i trained this model on the lambda cloud on an a100 gpu with about 10 million lines of text the total budget was less than 5 dollars
	Out: When I got home, I trained this model on the Lambda Cloud.
	Out: On an A100 GPU with about 10 million lines of text, the total budget was less than 5 dollars.

In: george hw bush was the president of the us for 8 years
	Out: George H.W. Bush was the president of the U.S. for 8 years.

In: i saw mr smith at the store he was shopping for a new lawn mower i suggested he get one of those new battery operated ones they're so much quieter
	Out: I saw Mr. Smith at the store he was shopping for a new lawn mower.
	Out: I suggested he get one of those new battery operated ones.
	Out: They're so much quieter.

In: i went to the fgw store and bought a new tg optical scope
	Out: I went to the FGW store and bought a new TG optical scope.

In: it's that man again itma was a radio comedy programme that was broadcast by the bbc for twelve series from 1939 to 1949 featuring tommy handley in the central role itma was a character driven comedy whose satirical targets included officialdom and the proliferation of minor wartime regulations parts of the scripts were rewritten in the hours before the broadcast to ensure topicality
	Out: It's that man again.
	Out: ITMA was a radio comedy programme that was broadcast by the BBC for Twelve Series from 1939 to 1949, featuring Tommy Handley.
	Out: In the central role, ITMA was a character driven comedy whose satirical targets included officialdom and the proliferation of minor wartime regulations.
	Out: Parts of the scripts were rewritten in the hours before the broadcast to ensure topicality.

```

</details>
    
# Model Details

This model implements the graph shown below, with brief descriptions for each step following.

![graph.png](https://s3.amazonaws.com/moonup/production/uploads/1678575121699-62d34c813eebd640a4f97587.png)


1. **Encoding**:
The model begins by tokenizing the text with a subword tokenizer.
The tokenizer used here is a `SentencePiece` model with a vocabulary size of 32k.
Next, the input sequence is encoded with a base-sized Transformer, consisting of 6 layers with a model dimension of 512.

2. **Punctuation**:
The encoded sequence is then fed into a feed-forward classification network to predict punctuation tokens. 
Punctation is predicted once per subword, to allow acronyms to be properly punctuated.
An indiret benefit of per-subword prediction is to allow the model to run in a graph generalized for continuous-script languages, e.g., Chinese.

5. **Sentence boundary detection**
For sentence boundary detection, we condition the model on punctuation via embeddings.
Each punctuation prediction is used to select an embedding for that token, which is concatenated to the encoded representation.
The SBD head analyzes both the encoding of the un-punctuated sequence and the puncutation predictions, and predicts which tokens are sentence boundaries. 

7. **Shift and concat sentence boundaries**
In English, the first character of each sentence should be upper-cased.
Thus, we should feed the sentence boundary information to the true-case classification network.
Since the true-case classification network is feed-forward and has no temporal context, each time step must embed whether it is the first word of a sentence.
Therefore, we shift the binary sentence boundary decisions to the right by one: if token `N-1` is a sentence boundary, token `N` is the first word of a sentence.
Concatenating this with the encoded text, each time step contains whether it is the first word of a sentence as predicted by the SBD head.

8. **True-case prediction**
Armed with the knowledge of punctation and sentence boundaries, a classification network predicts true-casing.
Since true-casing should be done on a per-character basis, the classification network makes `N` predictions per token, where `N` is the length of the subtoken.
(In practice, `N` is the longest possible subword, and the extra predictions are ignored).
This scheme captures acronyms, e.g., "NATO", as well as bi-capitalized words, e.g., "MacDonald".

The model's maximum length is 256 subtokens, due to the limit of the trained embeddings. 
However, the [punctuators](https://github.com/1-800-BAD-CODE/punctuators) package
as described above will transparently predict on overlapping subgsegments of long inputs and fuse the results before returning output,
allowing inputs to be arbitrarily long.

## Punctuation Tokens
This model predicts the following set of punctuation tokens:

| Token  | Description |
| ---: | :---------- |
| NULL    | Predict no punctuation |
| ACRONYM    | Every character in this subword ends with a period |
| .    | Latin full stop |
| ,    | Latin comma | 
| ?    | Latin question mark |

# Training Details

## Training Framework
This model was trained on a forked branch of the [NeMo](https://github.com/NVIDIA/NeMo) framework.

## Training Data
This model was trained with News Crawl data from WMT.

Approximately 10M lines were used from the years 2021 and 2012. 
The latter was used to attempt to reduce bias: annual news is typically dominated by a few topics, and 2021 is dominated by COVID discussions.

# Limitations
## Domain
This model was trained on news data, and may not perform well on conversational or informal data.

## Noisy Training Data
The training data was noisy, and no manual cleaning was utilized.

### Acronyms and Abbreviations
Acronyms and abbreviations are especially noisy; the table below shows how many variations of each token appear in the training data.

| Token  | Count |
| -: | :- |
| Mr    | 115232 |
| Mr.    | 108212 |

| Token  | Count |
| -: | :- |
| U.S.    | 85324 |
| US    | 37332 |
| U.S | 354 |
| U.s | 108 |
| u.S. | 65 |

Thus, the model's acronym and abbreviation predictions may be a bit unpredictable.

### Sentence Boundary Detection Targets
An assumption for sentence boundary detection targets is that each line of the input data is exactly one sentence.
However, a non-negligible portion of the training data contains multiple sentences per line.
Thus, the SBD head may miss an obvious sentence boundary if it's similar to an error seen in the training data.


# Evaluation
In these metrics, keep in mind that
1. The data is noisy
2. Sentence boundaries and true-casing are conditioned on predicted punctuation, which is the most difficult task and sometimes incorrect.
   When conditioning on reference punctuation, true-casing and SBD metrics are much higher w.r.t. the reference targets.
4. Punctuation can be subjective. E.g.,
   
   `Hello Frank, how's it going?`
   
   or

   `Hello Frank. How's it going?`

   When the sentences are longer and more practical, these ambiguities abound and affect all 3 analytics.

## Test Data and Example Generation
Each test example was generated using the following procedure:

1. Concatenate 10 random sentences
2. Lower-case the concatenated sentence
3. Remove all punctuation

The data is a held-out portion of News Crawl, which has been deduplicated. 
3,000 lines of data was used, generating 3,000 unique examples of 10 sentences each.

## Results

<details open>

  <summary>Punctuation Report</summary>

```text
    label                                                precision    recall       f1           support   
    <NULL> (label_id: 0)                                    98.83      98.49      98.66     446496
    <ACRONYM> (label_id: 1)                                 74.15      94.26      83.01        697
    . (label_id: 2)                                         90.64      92.99      91.80      30002
    , (label_id: 3)                                         77.19      79.13      78.15      23321
    ? (label_id: 4)                                         76.58      74.56      75.56       1022
    -------------------
    micro avg                                               97.21      97.21      97.21     501538
    macro avg                                               83.48      87.89      85.44     501538
    weighted avg                                            97.25      97.21      97.23     501538

```

</details>

<details open>

  <summary>True-casing Report</summary>

```text
# With predicted punctuation (not aligned with targets)
    label                                                precision    recall       f1           support   
    LOWER (label_id: 0)                                     99.76      99.72      99.74    2020678
    UPPER (label_id: 1)                                     93.32      94.20      93.76      83873
    -------------------
    micro avg                                               99.50      99.50      99.50    2104551
    macro avg                                               96.54      96.96      96.75    2104551
    weighted avg                                            99.50      99.50      99.50    2104551


# With reference punctuation (punctuation matches targets)
    label                                                precision    recall       f1           support   
    LOWER (label_id: 0)                                     99.83      99.81      99.82    2020678
    UPPER (label_id: 1)                                     95.51      95.90      95.71      83873
    -------------------
    micro avg                                               99.66      99.66      99.66    2104551
    macro avg                                               97.67      97.86      97.76    2104551
    weighted avg                                            99.66      99.66      99.66    2104551

```

</details>

<details open>

  <summary>Sentence Boundary Detection report</summary>

```text
# With predicted punctuation (not aligned with targets)
    label                                                precision    recall       f1           support   
    NOSTOP (label_id: 0)                                    99.59      99.45      99.52     471608
    FULLSTOP (label_id: 1)                                  91.47      93.53      92.49      29930
    -------------------
    micro avg                                               99.09      99.09      99.09     501538
    macro avg                                               95.53      96.49      96.00     501538
    weighted avg                                            99.10      99.09      99.10     501538


# With reference punctuation (punctuation matches targets)
    label                                                precision    recall       f1           support   
    NOSTOP (label_id: 0)                                   100.00      99.97      99.98     471608
    FULLSTOP (label_id: 1)                                  99.63      99.93      99.78      32923
    -------------------
    micro avg                                               99.97      99.97      99.97     504531
    macro avg                                               99.81      99.95      99.88     504531
    weighted avg                                            99.97      99.97      99.97     504531
    
```

</details>


# Fun Facts
Some fun facts are examined in this section.

## Embeddings
Let's examine the embeddings (see graph above) to see if the model meaningfully employed them.

We show here the cosine similarity between the embeddings of each token: 

| | NULL | ACRONYM | . | , | ? |
| - | - | - | - | - | - |
| NULL |	1.00	| | | | |
| ACRONYM |	-0.49 |	1.00  | | ||
| . |	-1.00 |	0.48 |	1.00 |	| |
| ,	| 1.00 |	-0.48 |	-1.00 |	1.00 |	|
| ?	| -1.00 |	0.49 |	1.00 |	-1.00 |	1.00 |	

Recall that these embeddings are used to predict sentence boundaries... thus we should expect full stops to cluster.

Indeed, we see that `NULL` and "`,`" are exactly the same, because neither have an implication on sentence boundaries.

Next, we see that "`.`" and "`?`" are exactly the same, because w.r.t. SBD these are exactly the same: strong full stop implications.
(Though, we may expect some difference between these tokens, given that "`.`" is predicted after abbreviations, e.g., 'Mr.', that are not full stops.)

Further, we see that "`.`" and "`?`" are exactly the opposite of `NULL`. 
This is expected since these tokens typically imply sentence boundaries, whereas `NULL` and "`,`" never do.

Lastly, we see that `ACRONYM` is similar to, but not the same as, the full stops "`.`" and "`?`",
and far from, but not the opposite of, `NULL` and "`,`".
Intuition suggests this is because acronyms can be full stops ("I live in the northern U.S. It's cold here.") or not ("It's 5 a.m. and I'm tired.").

