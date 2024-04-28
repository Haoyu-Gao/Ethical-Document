---
license:
- apache-2.0
- bsd-3-clause
tags:
- summarization
- led
- summary
- longformer
- booksum
- long-document
- long-form
datasets:
- kmfoda/booksum
metrics:
- rouge
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
- text: ' the big variety of data coming from diverse sources is one of the key properties
    of the big data phenomenon. It is, therefore, beneficial to understand how data
    is generated in various environments and scenarios, before looking at what should
    be done with this data and how to design the best possible architecture to accomplish
    this The evolution of IT architectures, described in Chapter 2, means that the
    data is no longer processed by a few big monolith systems, but rather by a group
    of services In parallel to the processing layer, the underlying data storage has
    also changed and became more distributed This, in turn, required a significant
    paradigm shift as the traditional approach to transactions (ACID) could no longer
    be supported. On top of this, cloud computing is becoming a major approach with
    the benefits of reducing costs and providing on-demand scalability but at the
    same time introducing concerns about privacy, data ownership, etc In the meantime
    the Internet continues its exponential growth: Every day both structured and unstructured
    data is published and available for processing: To achieve competitive advantage
    companies have to relate their corporate resources to external services, e.g.
    financial markets, weather forecasts, social media, etc While several of the sites
    provide some sort of API to access the data in a more orderly fashion; countless
    sources require advanced web mining and Natural Language Processing (NLP) processing
    techniques: Advances in science push researchers to construct new instruments
    for observing the universe O conducting experiments to understand even better
    the laws of physics and other domains. Every year humans have at their disposal
    new telescopes, space probes, particle accelerators, etc These instruments generate
    huge streams of data, which need to be stored and analyzed. The constant drive
    for efficiency in the industry motivates the introduction of new automation techniques
    and process optimization: This could not be done without analyzing the precise
    data that describe these processes. As more and more human tasks are automated,
    machines provide rich data sets, which can be analyzed in real-time to drive efficiency
    to new levels. Finally, it is now evident that the growth of the Internet of Things
    is becoming a major source of data. More and more of the devices are equipped
    with significant computational power and can generate a continuous data stream
    from their sensors. In the subsequent sections of this chapter, we will look at
    the domains described above to see what they generate in terms of data sets. We
    will compare the volumes but will also look at what is characteristic and important
    from their respective points of view. 3.1 The Internet is undoubtedly the largest
    database ever created by humans. While several well described; cleaned, and structured
    data sets have been made available through this medium, most of the resources
    are of an ambiguous, unstructured, incomplete or even erroneous nature. Still,
    several examples in the areas such as opinion mining, social media analysis, e-governance,
    etc, clearly show the potential lying in these resources. Those who can successfully
    mine and interpret the Internet data can gain unique insight and competitive advantage
    in their business An important area of data analytics on the edge of corporate
    IT and the Internet is Web Analytics.'
  example_title: data science textbook
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
- text: 'The majority of available text summarization datasets include short-form
    source documents that lack long-range causal and temporal dependencies, and often
    contain strong layout and stylistic biases. While relevant, such datasets will
    offer limited challenges for future generations of text summarization systems.
    We address these issues by introducing BookSum, a collection of datasets for long-form
    narrative summarization. Our dataset covers source documents from the literature
    domain, such as novels, plays and stories, and includes highly abstractive, human
    written summaries on three levels of granularity of increasing difficulty: paragraph-,
    chapter-, and book-level. The domain and structure of our dataset poses a unique
    set of challenges for summarization systems, which include: processing very long
    documents, non-trivial causal and temporal dependencies, and rich discourse structures.
    To facilitate future work, we trained and evaluated multiple extractive and abstractive
    summarization models as baselines for our dataset.'
  example_title: BookSum Abstract
inference:
  parameters:
    max_length: 96
    min_length: 8
    no_repeat_ngram_size: 3
    early_stopping: true
    repetition_penalty: 3.5
    length_penalty: 0.3
    encoder_no_repeat_ngram_size: 3
    num_beams: 4
model-index:
- name: pszemraj/led-base-book-summary
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: kmfoda/booksum
      type: kmfoda/booksum
      config: kmfoda--booksum
      split: test
    metrics:
    - type: rouge
      value: 33.4536
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmEzYjNkZTUxZjA0YTdmNTJkMjVkMTg2NDRjNTkzN2ZlNDlhNTBhMWQ5MTNiYWE4Mzg5YTMyMTM5YmZjNDI3OSIsInZlcnNpb24iOjF9.OWjM_HCQLQHK4AV4em70QGT3lrVk25WyZdcXA8ywest_XSx9KehJbsIMDKtXxOOMwxvkogKnScy4tbskYMQqDg
    - type: rouge
      value: 5.2232
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiOTVhOTdjZjc5YTdhMmVjZGE1NTA5MmJkYmM3Y2U3OGVlMjZmOGVlMTUzYTdiZGRhM2NmZjAzMjFkZjlkMzJmOCIsInZlcnNpb24iOjF9.qOlwWEe8dfBunmwImhbkcxzUW3ml-ESsuxjWN1fjn_o36zaUlDqlrXovMcL9GX9mVdvZDhx9W82rAR8h6410AQ
    - type: rouge
      value: 16.2044
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzkwOTEwYjkxYzlhMWE4ZjhlZDVjZWEwMWY2YzgwY2Q2YzJkYWFhMTQ4ODFlZmVkY2I1OWVhMTFmZThlOGY4NCIsInZlcnNpb24iOjF9.fJSr9wRQ07YIPMpb2_xv14EkHRz3gsPdZH-4LzpdviLOjVhlK1Y4gSZjp3PTEbu4Hua0umvNTMrhii8hp3DFBA
    - type: rouge
      value: 29.9765
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYWRkYjcwMTYwODRjN2E4MDliZWQyNjczNDU1NGZkMDRkNDlhNDA1YzZiOTk1MWJjZDkyMDg3MGMxYmVhOTA5MyIsInZlcnNpb24iOjF9.tUkVmhT0bl9eY_BzAzdzEI1lo3Iyfv6HBrrsVsRHqPFh4C0Q9Zk3IXbR-F_gMDx9vDiZIkpfG7SfsIZXwhDkBw
    - type: loss
      value: 3.1985862255096436
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiM2RmYzQ1NTFiYjk3YTZjMTI3NDJlMDY0MTgyZDZlZDRmZDcwOWE1YjU0OGYyZTJlY2RkZTEzZDFlNDk2ZjgyNSIsInZlcnNpb24iOjF9.Pc5Tfu8IXYeB5ETK2JMIL4gpRIvvYXVS6w1AZdfq9dD1dm9Te2xaNhzGBHviqgEfFI9APNSJB28wna1OpYP0Dg
    - type: gen_len
      value: 191.9783
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmMyMDI5MzFlNzNjODNmOWQ0ZTM3MzVkNTNkYzIxNTIwZDQzMTU2MTM0YjYzNjJiMGRhOTQ0OWFhN2U4N2NjYyIsInZlcnNpb24iOjF9.AfsX-O1YwfbPxUwAD7rd1Ub7SXth7FFpTo2iNSOUWFhYmDUECkf6qtJ5pVHXXZwnpidAlfPTPg-5y3dx_BBGCA
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
    metrics:
    - type: rouge
      value: 32
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmNhZjk3NjFlZDBhZjU2YzgzOTdhZTNkZjBkYjNjZDk2YjE2NDBmMDhiY2Y5M2EwNGI5Njk1NWU3ZDYyMzk2ZSIsInZlcnNpb24iOjF9.htkMQQLjIeFFjnpAJOwwxAdgzGZX10Und6RONubeeydXqQqb562EHqAw0K1ZlqltC4GBGKK3xslGOWXQ5AV6CA
    - type: rouge
      value: 10.0781
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWYzZDA1YmU5YTkzMjEwN2IzMTNhZmZmOTU2ZGUyNzdlNWQ0OGQ1Y2UxOGQ0NWUyOWVmZmZkYzFkODE3OTliNiIsInZlcnNpb24iOjF9.WVE3fmYLkOW32_neYYj4TNJ5lhrG-27DnoJd4YDUzpHYvGWGoFU9CUuIFraQFnojRr02f3KqVY7T33DG5mpzBg
    - type: rouge
      value: 23.6331
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYTYyOTE0ODY2Mjk0YTk5ZTY5NTZkM2JkOGZhNjQ3NjNiMjVhNTc4ZmMwYzg1ZGIxOTA2MDQxNmU3Yjc5YWY0MSIsInZlcnNpb24iOjF9.yQ8WpdsyGKSuTG8MxHXqujEAYOIrt_hoUbuHc8HnS-GjS9xJ-rKO6pP6HYbi0LC9Xqh2_QPveCpNqr9ZQMGRCg
    - type: rouge
      value: 28.7831
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzVkMDNlODA4NWI3OGI1OGFlNjFlNWE4YzY5ZDE1NDdhMjIwYjlkNDIxNDZjOGRiNTI1MGJkMmE0YWZiMDNhMiIsInZlcnNpb24iOjF9.qoxn2g70rbbX6sVCvm_cXzvYZf1UdTDU44vvEVdZL-4h36cJRCOx5--O1tZEVdyvlMVi-tYz1RSxLRwQd72FAw
    - type: loss
      value: 2.903024673461914
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZGM2M2NlY2Q3NjYxY2EyM2FkYmM5OGVhYzcyNjA3ZTFlYzc3M2M2ODNmNWVjNjZmMGNiODc4MWY5NWE2ZDMyNyIsInZlcnNpb24iOjF9.pC4UK75LbyVFFm0-fcStMtdQhbuHE37wkZHoVbSQOYSyxjI8yA46bQkPmgg5znby9FK_wIgGxC_4KOdEeN4jBw
    - type: gen_len
      value: 60.7411
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZWEwMDFiYjgyNzRhZDVmOWIzYzZlZWU5OTFkYmU4YzI2Mjk2OTg1ZDVlNzU0YzNhOWI1MmU2NTAxZWUzZmFlOCIsInZlcnNpb24iOjF9.Zepow4AFj1sQ6zyJGoy_Dl4ICKRtzZI2nVYWlTsDnGrBDT42ak9mFUuw-BjHR8dEVHJKmOZlLk6GJ09bL7tGAA
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: cnn_dailymail
      type: cnn_dailymail
      config: 3.0.0
      split: test
    metrics:
    - type: rouge
      value: 30.5036
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMmFkM2M4YTcyODEwMzY1MWViYTY0NmEzNjYwNGM4OTI4MmY1ZTk2ZjVjZjMwOGUwM2JiYTA0YjdkMWRkZTQ5MyIsInZlcnNpb24iOjF9.GatKuC1oPoD1HT9pA9lGAj6GNjhe3ADSNgZ5apntAFCHETlNV1mNf1zQ-rgFH2FP-lF3qS56Jn54pFp6FMwaBw
    - type: rouge
      value: 13.2558
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjUwZjBmMTUzNmM3ZTRjODQ0MGFiM2I3Y2ViMDRkODQzNGI3YzM0MmJiNzU1N2UwOTZmMGFkOTQwMzNjNmFiMSIsInZlcnNpb24iOjF9.kOWpg36sB5GdPVYUZpWlS0pSKu5mKmHcLmJO1I3oUzMSiwDeUpAPLXNC0u_gJMFaFdsaNTywepDuttLdB2oBBg
    - type: rouge
      value: 19.0284
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMTJmYzZmZWJiNTljYmJiZTllODk0NjdmNGNkZWZlZjMwMGE5YTAzMjMwNTcyNGM4MWE4MDUzYjM3NzQ5NzA2ZCIsInZlcnNpb24iOjF9.ooUqXvZC6ci_XxKrIcox2R2A0C8qyN0HP5djFMMb9SfoAaJAgdM0j6qsVQj9ccr0AgeRRIPNH_vI3gg-_lvaDw
    - type: rouge
      value: 28.3404
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTcxMDg5ZGI1MDRmNzM0ZmEyZmNiZGYxZTg0NzA4N2U0YTY3MGYxMjgzMzI0NjVlNWNiYTZmNWZjMzZkMmYzNiIsInZlcnNpb24iOjF9.RbEZQB2-IPb-l6Z1xeOE42NGwX1KQjlr2wNL9VH75L1gmMxKGTPMR_Yazma84ZKK-Ai7s2YPNh-MDanNU_4GCw
    - type: loss
      value: 3.9438512325286865
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZjQ2YmE1OTE5NDJlMTBhZGMzNDE5OThmNzMzOTRlYjEzMjc2ZDgyMDliNGY1NjFhOGQ0N2NkYmUzZGUwOGVlZiIsInZlcnNpb24iOjF9.FAwbzK-XJc-oEBFO7m8p4hkDCZDEhmU0ZSytrim-uHHcSFjRvbL-dF8rIvKVcxw5QeZ6QKZ7EkjDT7Ltt8KyCA
    - type: gen_len
      value: 231.0935
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiOTMzMTMyYjhhNjFiYjMyNDlhYzQzODM0MWNhNjkwMDVjNmFjYTk2NmQ4NzJlZjlhZjM2MGMwNWI1MjIxMGNiZCIsInZlcnNpb24iOjF9.mHDxhA2wVj6FDx7un4028-A8iGMFcPlSb5vH2DPGLPzQHBhSlvNac4-OELZf0PRmsXSb1nIqHqU-S_WUs8OSBg
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: billsum
      type: billsum
      config: default
      split: test
    metrics:
    - type: rouge
      value: 36.8502
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmE2ZjI4YmJkZGVjZDkzNzU5ZmI2MDYzNGZkNjE2OGM0Y2Y0Nzk1NTc1ZmUyZmFhYjIwY2RhMDVkMzQ1MWIxYyIsInZlcnNpb24iOjF9.SZjhhFkKwvRrI-Yl29psn17u1RCISsmmLVXxo2kxCjkhtMOma-EzC5YidjPDGQLb-J2nvqUworaC2pL_oeHxDQ
    - type: rouge
      value: 15.9147
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODgwOTJhOWIyZDQ4ZDA5YWMzYTJkZWFmMzlkNWYxNTg5OGFiNzY0MTExNTgyMTdlMTQ1N2EwYWY4OGZkNWY5YyIsInZlcnNpb24iOjF9.DS-X3eA1tGhVSuUL8uSPtJMNijODF3ugaKEtBglmPqF1OQZwIwQs-NExNYP4d6Y4Pa9d-DujD5yfyl9C8HBGCw
    - type: rouge
      value: 23.4762
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYTYxNTA4YzhmYTQ0YmRjMWU5ZDliZWFhMjM4ZmUyNGUyOWJhNzA1MDBhZDliYmYyYzY3NjBmZTZlYWY3YTY3ZCIsInZlcnNpb24iOjF9.o0W7dqdz0sqMPKtJbXSRpyVNsREEUypW-bGv7TW5lfJFkijfDKhVITEClFLWu5n2tIV-sXAYxgQHDf5_hpY-Dw
    - type: rouge
      value: 30.9597
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzEzOGNiYjk4NDkxNTFmMjA5YjM1YTQzZTk2N2JiZDgxNzAxYzFlYjliZjA3NmRjMzZlNGYyODBkNTI1NzVjNiIsInZlcnNpb24iOjF9.C_hobTR0ZY958oUZcGEKj2RoPOkyfMCTznwi4mUx-bfGRRAecMyn45bWVwwRq12glk1vThDetCjOMHA6jgSDCw
    - type: loss
      value: 3.878790855407715
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmYyOWM0YWQ0MjAxZDg5ZWQyNDk3MGUwNzdkOWIwZDc0OGJjYTU3YjZmOWY0YTljNDI0OWRlNTI0ZDMwZWEzOCIsInZlcnNpb24iOjF9.P01Jzfa-5jyMeoEqEsEluKOydNmtRtNy8YhwfJuYHVJTVDzCIfzY8b7iNfqTfKFKwKkZ4eTwmA6vmsPZeASDAw
    - type: gen_len
      value: 131.3622
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmJjN2Q5ZGNlZjQ2ODJiYTZlMzZmNWVmMzRlMGQ0ZTkxZWM3ZDQ4ZmQ1NmUyZjY4MTVhZGE5NDFiZTBhNDZiYSIsInZlcnNpb24iOjF9.DqYNc0ZCX_EqRi4zbSBAtb-js_JBHSWZkeGR9gSwEkJletKYFxPGZWd-B1ez88aj6PO775-qHd98xx3IWCHECQ
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: big_patent
      type: big_patent
      config: y
      split: test
    metrics:
    - type: rouge
      value: 33.7585
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiM2VmMGU5YWJlZWFlNjA3MDY2NTBmZWU3YWQxYTk3OGYzZmU5NmFmMTQ1NTVmNDQyZTJkNDMwY2E5NGRjMGU3MSIsInZlcnNpb24iOjF9.P6Rt9c3Xi_B-u8B1ug4paeZDoAO4ErGeNM0gELHGeOMj4XMjeSvyAW_-30cA9Wf23-0jGPOSZbN5pME4JpxfDA
    - type: rouge
      value: 9.4101
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNDA0NzUxMjIwYTFjNGQ5YTA4YjE1NGU5YWMzYjhiOTk2NWE3ZGQxNDY4YTI3ZmI0ODBjYmJkZjcwYTM2OTg2MCIsInZlcnNpb24iOjF9.23hd2SuLoX3_Rygj2ykcSQccPeFsf4yLDAgvS189jx6JNln0MVR6YI2-3Yzo5g8LJk0MCbgkOp0my-nf7nMaDw
    - type: rouge
      value: 18.8927
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODhhMGZiZWFlNmZkYmYxZjJmODE1NWRiZjI2OGU1MTc4MDkyYjk1Mzk5ODFkYWVhY2ExNTViYjJmYzkzNWJhYiIsInZlcnNpb24iOjF9.SkKhf-l2cl2KcuC17oPrBtkBlZJaj2ujCgzRlfZy76rU9JtlW7N9bcy1ugnw-vRVUVVR6wUK08T45YorfuxqBg
    - type: rouge
      value: 28.5051
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMTgzYzA0NmQ0OTZmNzJkNGZiNTdmMzFmOTljMWE3YzM0NDg2MDY1ZDY5ZTE4MmQ5YzU1ZDFiNmE2ZjkwMjRjMiIsInZlcnNpb24iOjF9.p1TQINRxMatNe77_BMnusSg1K5FOD9f1_N4TBJDjJHNhYnyQDE4pKHfK8j6fsHGg58DHVQjmm8g96SK4uMF6DA
    - type: loss
      value: 5.162865161895752
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZWM1YTQ4MjVmMDkyZDI3OWJmODhmOWE2MDYyMDA4OGRmYzhiY2YzZjVmMTZkMTI4NjBlY2MwMDY3ZDE5ZjlmMyIsInZlcnNpb24iOjF9.Czh4TOG-QIqyc_-GJ3wc1TLuxc-KLwPelV5tiwEjNhZFyUZkjLH__ccOxBk9TYy2vunvh2AwdY3Mt6Fr8LhaDA
    - type: gen_len
      value: 222.6626
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2JjNzVkODhmOWQ5NWMwNDdlNzhkYjE5NjY3NTgwNWVmZDZlMzc4NDdmZjdlN2M2ODBkZGU5NGU0ZjMzM2Q5OCIsInZlcnNpb24iOjF9.z4hZ-uXg8PPn-THRHFrsWZpS3jgE8URk5yoLenwWtev5toTrZ2Y-DP8O30nPnzMkzA4yzo_NUKIACxoUdMqfCQ
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: multi_news
      type: multi_news
      config: default
      split: test
    metrics:
    - type: rouge
      value: 38.7332
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMGViMThhNTdlZDRiMTg5NTZjNGVmOThiMjI5NDEyZDMxYjU4MTU2ZTliZjZmMzAzMmRhNDIxYjViYjZmNWYwNSIsInZlcnNpb24iOjF9.SK_1Q9WlkNhu3mfsyir1l72pddjURZvJV3mcJ4jhBxS2k2q1NAR8JT_iT8v1thLiv8NUDmDr2o9Dig4A8svDBw
    - type: rouge
      value: 11.0072
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzkzMDU1ZGZlOWUwOGQyY2UwMWFjZTY1MDBmNzcyZGYzZTliNGVkNDZjZDVjZjA4NmE3OWVhMGIyZmE3NGE0NSIsInZlcnNpb24iOjF9.j0wvR0NPw0lqxW3ASbmBvxAbFHGikXw-Y7FjutojhzTfSs3BIs5Z8s5_h6eesvSGT5fS_qUrbnl9EEBwjrXqDg
    - type: rouge
      value: 18.6018
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjIwNTUzN2ZhZjU5OGFhYzRmZmEwY2NkZWVjYmYzZjRjMGIxNzNjZDY5YzIyMTg2NDJkMGYxYmViNTcwOTc5NCIsInZlcnNpb24iOjF9.rD_tFYRyb-o6VX7Z52fULvP_HQjqqshqnvbjAxWjuCM9hCn1J6oh0zAASPw0k1lWiURbiMCiaxIHxe_5BN_rAQ
    - type: rouge
      value: 34.5911
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2Q4MWY3NGFhNjE5YjE5NzIyODVhNTYxNWFmZDE5NjNiZTM1M2M3ZmIwNTZiOWEyMTc2MzQ0MWQ5YTdjYThlNyIsInZlcnNpb24iOjF9.R789HgYsv_k6OrjocVi0ywx0aCRlgOKpEWUiSUDca-AfoDS8ADJBtLYoEKg1wnRlR9yWoD4vtEWdKbyOOln1CA
    - type: loss
      value: 3.5744354724884033
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzBjZTk0YWMwMzQxNDRlY2UxZDc4NTE1MmEzNDkwM2M3ZGZhNGMzNmI4ZDU2ZTVhZDkwMjNhYTkxZTIwN2E4MyIsInZlcnNpb24iOjF9.bDQ_3-CumosWKroMwBEMwKnDAj4ENQbUnbS387hU0zAY1K5g1NOy7fKBohxYZnRVolEfiuhszifUMW9zcLjqCA
    - type: gen_len
      value: 192.0014
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNDQxZmEwYmU5MGI1ZWE5NTIyMmM1MTVlMjVjNTg4MDQyMjJhNGE5NDJhNmZiN2Y4ZDc4ZmExNjBkMjQzMjQxMyIsInZlcnNpb24iOjF9.o3WblPY-iL1vT66xPwyyi1VMPhI53qs9GJ5HsHGbglOALwZT4n2-6IRxRNcL2lLj9qUehWUKkhruUyDM5-4RBg
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: xsum
      type: xsum
      config: default
      split: test
    metrics:
    - type: rouge
      value: 16.3186
      name: ROUGE-1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYjNiYzkxNTc1M2ZiYzY4NmVhY2U4MGU0YWE1NzQ4YzQxNjM1ZThmOWU3ZjUwMWUxMWM1NTQyYzc0OWQ5MzQyZSIsInZlcnNpb24iOjF9.cDZzbzxrXaM4n-Fa-vBpUgq7ildtHg9hlO5p9pt58VYLGK3rsid3oUE2qsFH6Qk63j2cF4_hzgq93xoVlnR3Dg
    - type: rouge
      value: 3.0261
      name: ROUGE-2
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYjkzNzA0ODk3NWJjOGM2ZWFlY2MyZWM4NzZlYzZiMGQ2ODc0NzgzNDYzYmVlZjg2ZjBmNDMwOGViYTljYWQ2NSIsInZlcnNpb24iOjF9.ohBfAUhEktfITK6j_NusN5SOmF4XUHZWPNMpGrsGXRHTf1bUl6_UEQ0S3w58WQsgIuV3MkxWNRBU1oZAm3fbBQ
    - type: rouge
      value: 10.4045
      name: ROUGE-L
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMDM2ZDZhYzBiNGM3NDdhODlmNjJhMTNlZDE3ZTZmYjM1MWU5YmE0ODMyZGFhMmM0YmMwMzNiZWU4ZDAzMDFlNiIsInZlcnNpb24iOjF9.653PFaov_0t8g_fVyVxm8DBx7uV4646yK0rtxOxC7qsnRdljdThSOklw9tND5-44WdkzipzuLyVzq1qe-TbKBA
    - type: rouge
      value: 12.612
      name: ROUGE-LSUM
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmY5YzU2ZjE2OWM0ZGQwZmVjZjQwZTQ0MDNkZmNiMTdhZjFkMDA5OGFhYWQ0Y2QwZDY0YWJlNWUxZGQ0YTUwZiIsInZlcnNpb24iOjF9.RXyu1jIj_gV26WCHSGHZufWXKFEexuRaLD4gkOvlBcaXJrFoE11tttB6mYzN6Tk8qx5cvV5L_ZIUfDmOqunkAA
    - type: loss
      value: 3.323798179626465
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZjU5ZWUxMjIwMWYwNDY1YzUwMzUxNGFiZWI3ZDVhZDFlYzJhNzk3MjA1OGExNTg0NjZlOGQyYzBiZjdhN2E2YSIsInZlcnNpb24iOjF9.vFxH1vHAACKE4XcgBhuoaV38yUZuYJuNm23V3nWVbF4FwyN79srV3Y9CqPGoOiIoUSQJ9fdKZXZub5j0GuUJAA
    - type: gen_len
      value: 149.7551
      name: gen_len
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzg1ZjY5MTJkMTgzMjhiYzMxNjkyZjlmNmI2ZGU0YTRhZjU5NjQwOWE5MjczZDIxNGI1MGI4YzhhOGVkZDFkYSIsInZlcnNpb24iOjF9.S7W5-vqldJuqtC5MweC3iCK6uy-uTRe4kGqoApMl2Sn6w9sVHnY7u905yNLXzFLrLYMgjlct5LB7AAirHeEJBw
---
# LED-Based Summarization Model: Condensing Long and Technical Information

<a href="https://colab.research.google.com/gist/pszemraj/36950064ca76161d9d258e5cdbfa6833/led-base-demo-token-batching.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The Longformer Encoder-Decoder (LED) for Narrative-Esque Long Text Summarization is a model I fine-tuned from [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) to condense extensive technical, academic, and narrative content in a fairly generalizable way.

## Key Features and Use Cases

- Ideal for summarizing long narratives, articles, papers, textbooks, and other documents.
  - the sparknotes-esque style leads to 'explanations' in the summarized content, offering insightful output.
- High capacity: Handles up to 16,384 tokens per batch.
- demos: try it out in the notebook linked above or in the [demo on Spaces](https://huggingface.co/spaces/pszemraj/summarize-long-text)

> **Note:** The API widget has a max length of ~96 tokens due to inference timeout constraints. 

## Training Details

The model was trained on the BookSum dataset released by SalesForce, which leads to the `bsd-3-clause` license. The training process involved 16 epochs with parameters tweaked to facilitate very fine-tuning-type training (super low learning rate). 

Model checkpoint: [`pszemraj/led-base-16384-finetuned-booksum`](https://huggingface.co/pszemraj/led-base-16384-finetuned-booksum). 

## Other Related Checkpoints

This model is the smallest/fastest booksum-tuned model I have worked on. If you're looking for higher quality summaries, check out:

- [Long-T5-tglobal-base](https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary)
- [BigBird-Pegasus-Large-K](https://huggingface.co/pszemraj/bigbird-pegasus-large-K-booksum)
- [Pegasus-X-Large](https://huggingface.co/pszemraj/pegasus-x-large-book-summary)
- [Long-T5-tglobal-XL](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary)

There are also other variants on other datasets etc on my hf profile, feel free to try them out :)

---

## Basic Usage

I recommend using `encoder_no_repeat_ngram_size=3` when calling the pipeline object, as it enhances the summary quality by encouraging the use of new vocabulary and crafting an abstractive summary.

Create the pipeline object:

```python
import torch
from transformers import pipeline

hf_name = "pszemraj/led-base-book-summary"

summarizer = pipeline(
    "summarization",
    hf_name,
    device=0 if torch.cuda.is_available() else -1,
)
```

Feed the text into the pipeline object:

```python
wall_of_text = "your words here"

result = summarizer(
    wall_of_text,
    min_length=8,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    do_sample=False,
    early_stopping=True,
)
print(result[0]["generated_text"])
```

## Simplified Usage with TextSum

To streamline the process of using this and other models, I've developed [a Python package utility](https://github.com/pszemraj/textsum) named `textsum`. This package offers simple interfaces for applying summarization models to text documents of arbitrary length. 

Install TextSum:

```bash
pip install textsum
```

Then use it in Python with this model:

```python
from textsum.summarize import Summarizer

model_name = "pszemraj/led-base-book-summary"
summarizer = Summarizer(
    model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
    token_batch_length=4096,  # how many tokens to batch summarize at a time
)
long_string = "This is a long string of text that will be summarized."
out_str = summarizer.summarize_string(long_string)
print(f"summary: {out_str}")
```

Currently implemented interfaces include a Python API, a Command-Line Interface (CLI), and a shareable demo/web UI. 

For detailed explanations and documentation, check the [README](https://github.com/pszemraj/textsum) or the [wiki](https://github.com/pszemraj/textsum/wiki)

---