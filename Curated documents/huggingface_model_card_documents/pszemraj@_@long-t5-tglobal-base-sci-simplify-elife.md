---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- lay summaries
- paper summaries
- biology
- medical
datasets:
- pszemraj/scientific_lay_summarisation-elife-norm
widget:
- text: large earthquakes along a given fault segment do not occur at random intervals
    because it takes time to accumulate the strain energy for the rupture. The rates
    at which tectonic plates move and accumulate strain at their boundaries are approximately
    uniform. Therefore, in first approximation, one may expect that large ruptures
    of the same fault segment will occur at approximately constant time intervals.
    If subsequent main shocks have different amounts of slip across the fault, then
    the recurrence time may vary, and the basic idea of periodic mainshocks must be
    modified. For great plate boundary ruptures the length and slip often vary by
    a factor of 2. Along the southern segment of the San Andreas fault the recurrence
    interval is 145 years with variations of several decades. The smaller the standard
    deviation of the average recurrence interval, the more specific could be the long
    term prediction of a future mainshock.
  example_title: earthquakes
- text: ' A typical feed-forward neural field algorithm. Spatiotemporal coordinates
    are fed into a neural network that predicts values in the reconstructed domain.
    Then, this domain is mapped to the sensor domain where sensor measurements are
    available as supervision. Class and Section Problems Addressed Generalization
    (Section 2) Inverse problems, ill-posed problems, editability; symmetries. Hybrid
    Representations (Section 3) Computation & memory efficiency, representation capacity,
    editability: Forward Maps (Section 4) Inverse problems Network Architecture (Section
    5) Spectral bias, integration & derivatives. Manipulating Neural Fields (Section
    6) Edit ability, constraints, regularization. Table 2: The five classes of techniques
    in the neural field toolbox each addresses problems that arise in learning, inference,
    and control. (Section 3). We can supervise reconstruction via differentiable forward
    maps that transform Or project our domain (e.g, 3D reconstruction via 2D images;
    Section 4) With appropriate network architecture choices, we can overcome neural
    network spectral biases (blurriness) and efficiently compute derivatives and integrals
    (Section 5). Finally, we can manipulate neural fields to add constraints and regularizations,
    and to achieve editable representations (Section 6). Collectively, these classes
    constitute a ''toolbox'' of techniques to help solve problems with neural fields
    There are three components in a conditional neural field: (1) An encoder or inference
    function € that outputs the conditioning latent variable 2 given an observation
    0 E(0) =2. 2 is typically a low-dimensional vector, and is often referred to aS
    a latent code Or feature code_ (2) A mapping function 4 between Z and neural field
    parameters O: Y(z) = O; (3) The neural field itself $. The encoder € finds the
    most probable z given the observations O: argmaxz P(2/0). The decoder maximizes
    the inverse conditional probability to find the most probable 0 given Z: arg-
    max P(Olz). We discuss different encoding schemes with different optimality guarantees
    (Section 2.1.1), both global and local conditioning (Section 2.1.2), and different
    mapping functions Y (Section 2.1.3) 2. Generalization Suppose we wish to estimate
    a plausible 3D surface shape given a partial or noisy point cloud. We need a suitable
    prior over the sur- face in its reconstruction domain to generalize to the partial
    observations. A neural network expresses a prior via the function space of its
    architecture and parameters 0, and generalization is influenced by the inductive
    bias of this function space (Section 5).'
  example_title: scientific paper
- text: 'Is a else or outside the cob and tree written being of early client rope
    and you have is for good reasons. On to the ocean in Orange for time. By''s the
    aggregate we can bed it yet. Why this please pick up on a sort is do and also
    M Getoi''s nerocos and do rain become you to let so is his brother is made in
    use and Mjulia''s''s the lay major is aging Masastup coin present sea only of
    Oosii rooms set to you We do er do we easy this private oliiishs lonthen might
    be okay. Good afternoon everybody. Welcome to this lecture of Computational Statistics.
    As you can see, I''m not socially my name is Michael Zelinger. I''m one of the
    task for this class and you might have already seen me in the first lecture where
    I made a quick appearance. I''m also going to give the tortillas in the last third
    of this course. So to give you a little bit about me, I''m a old student here
    with better Bulman and my research centres on casual inference applied to biomedical
    disasters, so that could be genomics or that could be hospital data. If any of
    you is interested in writing a bachelor thesis, a semester paper may be mastathesis
    about this topic feel for reach out to me. you have my name on models and my email
    address you can find in the directory I''d Be very happy to talk about it. you
    do not need to be sure about it, we can just have a chat. So with that said, let''s
    get on with the lecture. There''s an exciting topic today I''m going to start
    by sharing some slides with you and later on during the lecture we''ll move to
    the paper. So bear with me for a few seconds. Well, the projector is starting
    up. Okay, so let''s get started. Today''s topic is a very important one. It''s
    about a technique which really forms one of the fundamentals of data science,
    machine learning, and any sort of modern statistics. It''s called cross validation.
    I know you really want to understand this topic I Want you to understand this
    and frankly, nobody''s gonna leave Professor Mineshousen''s class without understanding
    cross validation. So to set the stage for this, I Want to introduce you to the
    validation problem in computational statistics. So the problem is the following:
    You trained a model on available data. You fitted your model, but you know the
    training data you got could always have been different and some data from the
    environment. Maybe it''s a random process. You do not really know what it is,
    but you know that somebody else who gets a different batch of data from the same
    environment they would get slightly different training data and you do not care
    that your method performs as well. On this training data. you want to to perform
    well on other data that you have not seen other data from the same environment.
    So in other words, the validation problem is you want to quantify the performance
    of your model on data that you have not seen. So how is this even possible? How
    could you possibly measure the performance on data that you do not know The solution
    to? This is the following realization is that given that you have a bunch of data,
    you were in charge. You get to control how much that your model sees. It works
    in the following way: You can hide data firms model. Let''s say you have a training
    data set which is a bunch of doubtless so X eyes are the features those are typically
    hide and national vector. It''s got more than one dimension for sure. And the
    why why eyes. Those are the labels for supervised learning. As you''ve seen before,
    it''s the same set up as we have in regression. And so you have this training
    data and now you choose that you only use some of those data to fit your model.
    You''re not going to use everything, you only use some of it the other part you
    hide from your model. And then you can use this hidden data to do validation from
    the point of you of your model. This hidden data is complete by unseen. In other
    words, we solve our problem of validation.'
  example_title: transcribed audio - lecture
- text: 'Transformer-based models have shown to be very useful for many NLP tasks.
    However, a major limitation of transformers-based models is its O(n^2)O(n 2) time
    & memory complexity (where nn is sequence length). Hence, it''s computationally
    very expensive to apply transformer-based models on long sequences n > 512n>512.
    Several recent papers, e.g. Longformer, Performer, Reformer, Clustered attention
    try to remedy this problem by approximating the full attention matrix. You can
    checkout 🤗''s recent blog post in case you are unfamiliar with these models.

    BigBird (introduced in paper) is one of such recent models to address this issue.
    BigBird relies on block sparse attention instead of normal attention (i.e. BERT''s
    attention) and can handle sequences up to a length of 4096 at a much lower computational
    cost compared to BERT. It has achieved SOTA on various tasks involving very long
    sequences such as long documents summarization, question-answering with long contexts.

    BigBird RoBERTa-like model is now available in 🤗Transformers. The goal of this
    post is to give the reader an in-depth understanding of big bird implementation
    & ease one''s life in using BigBird with 🤗Transformers. But, before going into
    more depth, it is important to remember that the BigBird''s attention is an approximation
    of BERT''s full attention and therefore does not strive to be better than BERT''s
    full attention, but rather to be more efficient. It simply allows to apply transformer-based
    models to much longer sequences since BERT''s quadratic memory requirement quickly
    becomes unbearable. Simply put, if we would have ∞ compute & ∞ time, BERT''s attention
    would be preferred over block sparse attention (which we are going to discuss
    in this post).

    If you wonder why we need more compute when working with longer sequences, this
    blog post is just right for you!

    Some of the main questions one might have when working with standard BERT-like
    attention include:

    Do all tokens really have to attend to all other tokens? Why not compute attention
    only over important tokens? How to decide what tokens are important? How to attend
    to just a few tokens in a very efficient way? In this blog post, we will try to
    answer those questions.

    What tokens should be attended to? We will give a practical example of how attention
    works by considering the sentence ''BigBird is now available in HuggingFace for
    extractive question answering''. In BERT-like attention, every word would simply
    attend to all other tokens.

    Let''s think about a sensible choice of key tokens that a queried token actually
    only should attend to by writing some pseudo-code. Will will assume that the token
    available is queried and build a sensible list of key tokens to attend to.

    >>> # let''s consider following sentence as an example >>> example = [''BigBird'',
    ''is'', ''now'', ''available'', ''in'', ''HuggingFace'', ''for'', ''extractive'',
    ''question'', ''answering'']

    >>> # further let''s assume, we''re trying to understand the representation of
    ''available'' i.e. >>> query_token = ''available'' >>> # We will initialize an
    empty `set` and fill up the tokens of our interest as we proceed in this section.
    >>> key_tokens = [] # => currently ''available'' token doesn''t have anything
    to attend Nearby tokens should be important because, in a sentence (sequence of
    words), the current word is highly dependent on neighboring past & future tokens.
    This intuition is the idea behind the concept of sliding attention.'
  example_title: bigbird blog intro
- text: 'To be fair, you have to have a very high IQ to understand Rick and Morty.
    The humour is extremely subtle, and without a solid grasp of theoretical physics
    most of the jokes will go over a typical viewer''s head. There''s also Rick''s
    nihilistic outlook, which is deftly woven into his characterisation- his personal
    philosophy draws heavily from Narodnaya Volya literature, for instance. The fans
    understand this stuff; they have the intellectual capacity to truly appreciate
    the depths of these jokes, to realise that they''re not just funny- they say something
    deep about LIFE. As a consequence people who dislike Rick & Morty truly ARE idiots-
    of course they wouldn''t appreciate, for instance, the humour in Rick''s existential
    catchphrase ''Wubba Lubba Dub Dub,'' which itself is a cryptic reference to Turgenev''s
    Russian epic Fathers and Sons. I''m smirking right now just imagining one of those
    addlepated simpletons scratching their heads in confusion as Dan Harmon''s genius
    wit unfolds itself on their television screens. What fools.. how I pity them.
    😂

    And yes, by the way, i DO have a Rick & Morty tattoo. And no, you cannot see it.
    It''s for the ladies'' eyes only- and even then they have to demonstrate that
    they''re within 5 IQ points of my own (preferably lower) beforehand. Nothin personnel
    kid 😎'
  example_title: Richard & Mortimer
- text: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey
    building, and the tallest structure in Paris. Its base is square, measuring 125
    metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed
    the Washington Monument to become the tallest man-made structure in the world,
    a title it held for 41 years until the Chrysler Building in New York City was
    finished in 1930. It was the first structure to reach a height of 300 metres.
    Due to the addition of a broadcasting aerial at the top of the tower in 1957,
    it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters,
    the Eiffel Tower is the second tallest free-standing structure in France after
    the Millau Viaduct.
  example_title: eiffel
parameters:
  max_length: 64
  min_length: 8
  no_repeat_ngram_size: 3
  early_stopping: true
  repetition_penalty: 3.5
  encoder_no_repeat_ngram_size: 4
  length_penalty: 0.4
  num_beams: 4
pipeline_tag: summarization
base_model: google/long-t5-tglobal-base
---


# long-t5-tglobal-base-sci-simplify: elife subset

<a href="https://colab.research.google.com/gist/pszemraj/37a406059887a400afc1428d70374327/long-t5-tglobal-base-sci-simplify-elife-example-with-textsum.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Exploring how well long-document models trained on "lay summaries" of scientific papers generalize. 

> A lay summary is a summary of a research paper or scientific study that is written in plain language, without the use of technical jargon, and is designed to be easily understood by non-experts.

## Model description

This model is a fine-tuned version of [google/long-t5-tglobal-base](https://huggingface.co/google/long-t5-tglobal-base) on the `pszemraj/scientific_lay_summarisation-elife-norm` dataset.

- The variant trained on the PLOS subset can be found [here](https://huggingface.co/pszemraj/long-t5-tglobal-base-sci-simplify)

## Usage 

It's recommended to use this model with [beam search decoding](https://huggingface.co/docs/transformers/generation_strategies#beamsearch-decoding). If interested, you can also use the `textsum` util repo to have most of this abstracted out for you:


```bash
pip install -U textsum
```

```python
from textsum.summarize import Summarizer

model_name = "pszemraj/long-t5-tglobal-base-sci-simplify-elife"
summarizer = Summarizer(model_name) # GPU auto-detected
text = "put the text you don't want to read here"
summary = summarizer.summarize_string(text)
print(summary)
```

## Intended uses & limitations

- Ability to generalize outside of the dataset domain (pubmed/bioscience type papers) has to be evaluated.

## Training and evaluation data

The `elife` subset of the lay summaries dataset. Refer to `pszemraj/scientific_lay_summarisation-elife-norm`

## Training procedure


### Eval results

It achieves the following results on the evaluation set:
- Loss: 1.9990
- Rouge1: 38.5587
- Rouge2: 9.7336
- Rougel: 21.1974
- Rougelsum: 35.9333
- Gen Len: 392.7095

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0004
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 16
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.01
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2 | Rougel  | Rougelsum | Gen Len  |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:------:|:-------:|:---------:|:--------:|
| 2.2995        | 1.47  | 100  | 2.0175          | 35.2501 | 8.2121 | 20.4587 | 32.4494   | 439.7552 |
| 2.2171        | 2.94  | 200  | 1.9990          | 38.5587 | 9.7336 | 21.1974 | 35.9333   | 392.7095 |

