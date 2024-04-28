---
language:
- zh
- en
- de
- fr
- it
- nl
- pt
- pl
- ru
license: apache-2.0
library_name: transformers
tags:
- text2vec
- feature-extraction
- sentence-similarity
- transformers
- mteb
datasets:
- https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-multilingual-dataset
metrics:
- spearmanr
pipeline_tag: sentence-similarity
model-index:
- name: text2vec-base-multilingual
  results:
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (en)
      type: mteb/amazon_counterfactual
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 70.97014925373134
    - type: ap
      value: 33.95151328318672
    - type: f1
      value: 65.14740155705596
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (de)
      type: mteb/amazon_counterfactual
      config: de
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 68.69379014989293
    - type: ap
      value: 79.68277579733802
    - type: f1
      value: 66.54960052336921
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (en-ext)
      type: mteb/amazon_counterfactual
      config: en-ext
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 70.90704647676162
    - type: ap
      value: 20.747518928580437
    - type: f1
      value: 58.64365465884924
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (ja)
      type: mteb/amazon_counterfactual
      config: ja
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 61.605995717344754
    - type: ap
      value: 14.135974879487028
    - type: f1
      value: 49.980224800472136
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonPolarityClassification
      type: mteb/amazon_polarity
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 66.103375
    - type: ap
      value: 61.10087197664471
    - type: f1
      value: 65.75198509894145
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (en)
      type: mteb/amazon_reviews_multi
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 33.134
    - type: f1
      value: 32.7905397597083
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (de)
      type: mteb/amazon_reviews_multi
      config: de
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 33.388
    - type: f1
      value: 33.190561196873084
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (es)
      type: mteb/amazon_reviews_multi
      config: es
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 34.824
    - type: f1
      value: 34.297290157740726
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (fr)
      type: mteb/amazon_reviews_multi
      config: fr
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 33.449999999999996
    - type: f1
      value: 33.08017234412433
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (ja)
      type: mteb/amazon_reviews_multi
      config: ja
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 30.046
    - type: f1
      value: 29.857141661482228
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (zh)
      type: mteb/amazon_reviews_multi
      config: zh
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 32.522
    - type: f1
      value: 31.854699911472174
  - task:
      type: Clustering
    dataset:
      name: MTEB ArxivClusteringP2P
      type: mteb/arxiv-clustering-p2p
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 32.31918856561886
  - task:
      type: Clustering
    dataset:
      name: MTEB ArxivClusteringS2S
      type: mteb/arxiv-clustering-s2s
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 25.503481615956137
  - task:
      type: Reranking
    dataset:
      name: MTEB AskUbuntuDupQuestions
      type: mteb/askubuntudupquestions-reranking
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 57.91471462820568
    - type: mrr
      value: 71.82990370663501
  - task:
      type: STS
    dataset:
      name: MTEB BIOSSES
      type: mteb/biosses-sts
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 68.83853315193127
    - type: cos_sim_spearman
      value: 66.16174850417771
    - type: euclidean_pearson
      value: 56.65313897263153
    - type: euclidean_spearman
      value: 52.69156205876939
    - type: manhattan_pearson
      value: 56.97282154658304
    - type: manhattan_spearman
      value: 53.167476517261015
  - task:
      type: Classification
    dataset:
      name: MTEB Banking77Classification
      type: mteb/banking77
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 78.08441558441558
    - type: f1
      value: 77.99825264827898
  - task:
      type: Clustering
    dataset:
      name: MTEB BiorxivClusteringP2P
      type: mteb/biorxiv-clustering-p2p
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 28.98583420521256
  - task:
      type: Clustering
    dataset:
      name: MTEB BiorxivClusteringS2S
      type: mteb/biorxiv-clustering-s2s
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 23.195091778460892
  - task:
      type: Classification
    dataset:
      name: MTEB EmotionClassification
      type: mteb/emotion
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 43.35
    - type: f1
      value: 38.80269436557695
  - task:
      type: Classification
    dataset:
      name: MTEB ImdbClassification
      type: mteb/imdb
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 59.348
    - type: ap
      value: 55.75065220262251
    - type: f1
      value: 58.72117519082607
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (en)
      type: mteb/mtop_domain
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 81.04879160966712
    - type: f1
      value: 80.86889779192701
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (de)
      type: mteb/mtop_domain
      config: de
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 78.59397013243168
    - type: f1
      value: 77.09902761555972
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (es)
      type: mteb/mtop_domain
      config: es
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 79.24282855236824
    - type: f1
      value: 78.75883867079015
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (fr)
      type: mteb/mtop_domain
      config: fr
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 76.16661446915127
    - type: f1
      value: 76.30204722831901
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (hi)
      type: mteb/mtop_domain
      config: hi
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 78.74506991753317
    - type: f1
      value: 77.50560442779701
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (th)
      type: mteb/mtop_domain
      config: th
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 77.67088607594937
    - type: f1
      value: 77.21442956887493
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (en)
      type: mteb/mtop_intent
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 62.786137710898316
    - type: f1
      value: 46.23474201126368
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (de)
      type: mteb/mtop_intent
      config: de
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 55.285996055226825
    - type: f1
      value: 37.98039513682919
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (es)
      type: mteb/mtop_intent
      config: es
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 58.67911941294196
    - type: f1
      value: 40.541410807124954
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (fr)
      type: mteb/mtop_intent
      config: fr
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 53.257124960851854
    - type: f1
      value: 38.42982319259366
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (hi)
      type: mteb/mtop_intent
      config: hi
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 59.62352097525995
    - type: f1
      value: 41.28886486568534
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (th)
      type: mteb/mtop_intent
      config: th
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 58.799276672694404
    - type: f1
      value: 43.68379466247341
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (af)
      type: mteb/amazon_massive_intent
      config: af
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 45.42030934767989
    - type: f1
      value: 44.12201543566376
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (am)
      type: mteb/amazon_massive_intent
      config: am
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 37.67652992602556
    - type: f1
      value: 35.422091900843164
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ar)
      type: mteb/amazon_massive_intent
      config: ar
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 45.02353732347007
    - type: f1
      value: 41.852484084738194
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (az)
      type: mteb/amazon_massive_intent
      config: az
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 48.70880968392737
    - type: f1
      value: 46.904360615435046
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (bn)
      type: mteb/amazon_massive_intent
      config: bn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 43.78950907868191
    - type: f1
      value: 41.58872353920405
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (cy)
      type: mteb/amazon_massive_intent
      config: cy
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 28.759246805648957
    - type: f1
      value: 27.41182001374226
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (da)
      type: mteb/amazon_massive_intent
      config: da
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 56.74176193678547
    - type: f1
      value: 53.82727354182497
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (de)
      type: mteb/amazon_massive_intent
      config: de
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 51.55682582380632
    - type: f1
      value: 49.41963627941866
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (el)
      type: mteb/amazon_massive_intent
      config: el
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 56.46940147948891
    - type: f1
      value: 55.28178711367465
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (en)
      type: mteb/amazon_massive_intent
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 63.83322125084063
    - type: f1
      value: 61.836172900845554
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (es)
      type: mteb/amazon_massive_intent
      config: es
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.27505043712172
    - type: f1
      value: 57.642436374361154
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (fa)
      type: mteb/amazon_massive_intent
      config: fa
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.05178211163417
    - type: f1
      value: 56.858998820504056
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (fi)
      type: mteb/amazon_massive_intent
      config: fi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 57.357094821788834
    - type: f1
      value: 54.79711189260453
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (fr)
      type: mteb/amazon_massive_intent
      config: fr
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.79959650302623
    - type: f1
      value: 57.59158671719513
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (he)
      type: mteb/amazon_massive_intent
      config: he
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 51.1768661735037
    - type: f1
      value: 48.886397276270515
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (hi)
      type: mteb/amazon_massive_intent
      config: hi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 57.06455951580362
    - type: f1
      value: 55.01530952684585
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (hu)
      type: mteb/amazon_massive_intent
      config: hu
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.3591123066577
    - type: f1
      value: 55.9277783370191
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (hy)
      type: mteb/amazon_massive_intent
      config: hy
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 52.108271687962336
    - type: f1
      value: 51.195023400664596
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (id)
      type: mteb/amazon_massive_intent
      config: id
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.26832548755883
    - type: f1
      value: 56.60774065423401
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (is)
      type: mteb/amazon_massive_intent
      config: is
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 35.806993947545394
    - type: f1
      value: 34.290418953173294
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (it)
      type: mteb/amazon_massive_intent
      config: it
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.27841291190315
    - type: f1
      value: 56.9438998642419
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ja)
      type: mteb/amazon_massive_intent
      config: ja
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 60.78009414929389
    - type: f1
      value: 59.15780842483667
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (jv)
      type: mteb/amazon_massive_intent
      config: jv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 31.153328850033624
    - type: f1
      value: 30.11004596099605
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ka)
      type: mteb/amazon_massive_intent
      config: ka
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 44.50235373234701
    - type: f1
      value: 44.040585262624745
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (km)
      type: mteb/amazon_massive_intent
      config: km
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 40.99193006052455
    - type: f1
      value: 39.505480119272484
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (kn)
      type: mteb/amazon_massive_intent
      config: kn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 46.95696032279758
    - type: f1
      value: 43.093638940785326
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ko)
      type: mteb/amazon_massive_intent
      config: ko
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.73100201748486
    - type: f1
      value: 52.79750744404114
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (lv)
      type: mteb/amazon_massive_intent
      config: lv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.865501008742434
    - type: f1
      value: 53.64798408964839
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ml)
      type: mteb/amazon_massive_intent
      config: ml
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 47.891728312037664
    - type: f1
      value: 45.261229414636055
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (mn)
      type: mteb/amazon_massive_intent
      config: mn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 52.2259583053127
    - type: f1
      value: 50.5903419246987
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ms)
      type: mteb/amazon_massive_intent
      config: ms
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.277067921990586
    - type: f1
      value: 52.472042479965886
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (my)
      type: mteb/amazon_massive_intent
      config: my
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 51.95696032279757
    - type: f1
      value: 49.79330411854258
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (nb)
      type: mteb/amazon_massive_intent
      config: nb
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.63685272360457
    - type: f1
      value: 52.81267480650003
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (nl)
      type: mteb/amazon_massive_intent
      config: nl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.451916610625425
    - type: f1
      value: 57.34790386645091
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (pl)
      type: mteb/amazon_massive_intent
      config: pl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.91055817081372
    - type: f1
      value: 56.39195048528157
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (pt)
      type: mteb/amazon_massive_intent
      config: pt
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.84196368527236
    - type: f1
      value: 58.72244763127063
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ro)
      type: mteb/amazon_massive_intent
      config: ro
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 57.04102219233354
    - type: f1
      value: 55.67040186148946
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ru)
      type: mteb/amazon_massive_intent
      config: ru
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.01613987895091
    - type: f1
      value: 57.203949825484855
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sl)
      type: mteb/amazon_massive_intent
      config: sl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 56.35843981170141
    - type: f1
      value: 54.18656338999773
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sq)
      type: mteb/amazon_massive_intent
      config: sq
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 56.47948890383322
    - type: f1
      value: 54.772224557130954
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sv)
      type: mteb/amazon_massive_intent
      config: sv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.43981170141224
    - type: f1
      value: 56.09260971364242
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sw)
      type: mteb/amazon_massive_intent
      config: sw
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 33.9609952925353
    - type: f1
      value: 33.18853392353405
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ta)
      type: mteb/amazon_massive_intent
      config: ta
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 44.29388029589778
    - type: f1
      value: 41.51986533284474
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (te)
      type: mteb/amazon_massive_intent
      config: te
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 47.13517148621385
    - type: f1
      value: 43.94784138379624
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (th)
      type: mteb/amazon_massive_intent
      config: th
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 56.856086079354405
    - type: f1
      value: 56.618177384748456
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (tl)
      type: mteb/amazon_massive_intent
      config: tl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 35.35978480161398
    - type: f1
      value: 34.060680080365046
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (tr)
      type: mteb/amazon_massive_intent
      config: tr
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.630127774041696
    - type: f1
      value: 57.46288652988266
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ur)
      type: mteb/amazon_massive_intent
      config: ur
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 52.7908540685945
    - type: f1
      value: 51.46934239116157
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (vi)
      type: mteb/amazon_massive_intent
      config: vi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.6469401479489
    - type: f1
      value: 53.9903066185816
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (zh-CN)
      type: mteb/amazon_massive_intent
      config: zh-CN
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 60.85743106926698
    - type: f1
      value: 59.31579548450755
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (zh-TW)
      type: mteb/amazon_massive_intent
      config: zh-TW
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 57.46805648957633
    - type: f1
      value: 57.48469733657326
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (af)
      type: mteb/amazon_massive_scenario
      config: af
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 50.86415601882985
    - type: f1
      value: 49.41696672602645
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (am)
      type: mteb/amazon_massive_scenario
      config: am
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 41.183591123066584
    - type: f1
      value: 40.04563865770774
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ar)
      type: mteb/amazon_massive_scenario
      config: ar
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 50.08069939475455
    - type: f1
      value: 50.724800165846126
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (az)
      type: mteb/amazon_massive_scenario
      config: az
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 51.287827841291204
    - type: f1
      value: 50.72873776739851
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (bn)
      type: mteb/amazon_massive_scenario
      config: bn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 46.53328850033624
    - type: f1
      value: 45.93317866639667
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (cy)
      type: mteb/amazon_massive_scenario
      config: cy
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 34.347679892400805
    - type: f1
      value: 31.941581141280828
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (da)
      type: mteb/amazon_massive_scenario
      config: da
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.073301950235376
    - type: f1
      value: 62.228728940111054
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (de)
      type: mteb/amazon_massive_scenario
      config: de
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 56.398789509078675
    - type: f1
      value: 54.80778341609032
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (el)
      type: mteb/amazon_massive_scenario
      config: el
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 61.79892400806993
    - type: f1
      value: 60.69430756982446
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (en)
      type: mteb/amazon_massive_scenario
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 66.96368527236046
    - type: f1
      value: 66.5893927997656
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (es)
      type: mteb/amazon_massive_scenario
      config: es
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.21250840618695
    - type: f1
      value: 62.347177794128925
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (fa)
      type: mteb/amazon_massive_scenario
      config: fa
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.43779421654339
    - type: f1
      value: 61.307701312085605
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (fi)
      type: mteb/amazon_massive_scenario
      config: fi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 61.09952925353059
    - type: f1
      value: 60.313907927386914
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (fr)
      type: mteb/amazon_massive_scenario
      config: fr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.38601210490922
    - type: f1
      value: 63.05968938353488
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (he)
      type: mteb/amazon_massive_scenario
      config: he
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 56.2878278412912
    - type: f1
      value: 55.92927644838597
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (hi)
      type: mteb/amazon_massive_scenario
      config: hi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 60.62878278412912
    - type: f1
      value: 60.25299253652635
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (hu)
      type: mteb/amazon_massive_scenario
      config: hu
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.28850033624748
    - type: f1
      value: 62.77053246337031
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (hy)
      type: mteb/amazon_massive_scenario
      config: hy
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 54.875588433086754
    - type: f1
      value: 54.30717357279134
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (id)
      type: mteb/amazon_massive_scenario
      config: id
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 61.99394754539341
    - type: f1
      value: 61.73085530883037
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (is)
      type: mteb/amazon_massive_scenario
      config: is
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 38.581035642232685
    - type: f1
      value: 36.96287269695893
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (it)
      type: mteb/amazon_massive_scenario
      config: it
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.350369872225976
    - type: f1
      value: 61.807327324823966
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ja)
      type: mteb/amazon_massive_scenario
      config: ja
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 65.17148621385338
    - type: f1
      value: 65.29620144656751
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (jv)
      type: mteb/amazon_massive_scenario
      config: jv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 36.12642905178212
    - type: f1
      value: 35.334393048479484
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ka)
      type: mteb/amazon_massive_scenario
      config: ka
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 50.26899798251513
    - type: f1
      value: 49.041065960139434
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (km)
      type: mteb/amazon_massive_scenario
      config: km
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 44.24344317417619
    - type: f1
      value: 42.42177854872125
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (kn)
      type: mteb/amazon_massive_scenario
      config: kn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 47.370544720914594
    - type: f1
      value: 46.589722581465324
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ko)
      type: mteb/amazon_massive_scenario
      config: ko
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 58.89038332212508
    - type: f1
      value: 57.753607921990394
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (lv)
      type: mteb/amazon_massive_scenario
      config: lv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 56.506388702084756
    - type: f1
      value: 56.0485860423295
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ml)
      type: mteb/amazon_massive_scenario
      config: ml
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 50.06388702084734
    - type: f1
      value: 50.109364641824584
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (mn)
      type: mteb/amazon_massive_scenario
      config: mn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 55.053799596503026
    - type: f1
      value: 54.490665705666686
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ms)
      type: mteb/amazon_massive_scenario
      config: ms
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 59.77135171486213
    - type: f1
      value: 58.2808650158803
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (my)
      type: mteb/amazon_massive_scenario
      config: my
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 55.71620712844654
    - type: f1
      value: 53.863034882475304
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (nb)
      type: mteb/amazon_massive_scenario
      config: nb
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 60.26227303295225
    - type: f1
      value: 59.86604657147016
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (nl)
      type: mteb/amazon_massive_scenario
      config: nl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.3759246805649
    - type: f1
      value: 62.45257339288533
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (pl)
      type: mteb/amazon_massive_scenario
      config: pl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.552118359112306
    - type: f1
      value: 61.354449605776765
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (pt)
      type: mteb/amazon_massive_scenario
      config: pt
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.40753194351043
    - type: f1
      value: 61.98779889528889
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ro)
      type: mteb/amazon_massive_scenario
      config: ro
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 60.68258238063214
    - type: f1
      value: 60.59973978976571
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ru)
      type: mteb/amazon_massive_scenario
      config: ru
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.31002017484868
    - type: f1
      value: 62.412312268503655
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sl)
      type: mteb/amazon_massive_scenario
      config: sl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 61.429051782111635
    - type: f1
      value: 61.60095590401424
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sq)
      type: mteb/amazon_massive_scenario
      config: sq
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.229320780094156
    - type: f1
      value: 61.02251426747547
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sv)
      type: mteb/amazon_massive_scenario
      config: sv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 64.42501681237391
    - type: f1
      value: 63.461494430605235
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sw)
      type: mteb/amazon_massive_scenario
      config: sw
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 38.51714862138534
    - type: f1
      value: 37.12466722986362
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ta)
      type: mteb/amazon_massive_scenario
      config: ta
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 46.99731002017485
    - type: f1
      value: 45.859147049984834
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (te)
      type: mteb/amazon_massive_scenario
      config: te
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 51.01882985877605
    - type: f1
      value: 49.01040173136056
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (th)
      type: mteb/amazon_massive_scenario
      config: th
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.234700739744454
    - type: f1
      value: 62.732294595214746
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (tl)
      type: mteb/amazon_massive_scenario
      config: tl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 38.72225958305312
    - type: f1
      value: 36.603231928120906
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (tr)
      type: mteb/amazon_massive_scenario
      config: tr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 64.48554135843982
    - type: f1
      value: 63.97380562022752
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ur)
      type: mteb/amazon_massive_scenario
      config: ur
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 56.7955615332885
    - type: f1
      value: 55.95308241204802
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (vi)
      type: mteb/amazon_massive_scenario
      config: vi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 57.06455951580362
    - type: f1
      value: 56.95570494066693
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (zh-CN)
      type: mteb/amazon_massive_scenario
      config: zh-CN
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 65.8338937457969
    - type: f1
      value: 65.6778746906008
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (zh-TW)
      type: mteb/amazon_massive_scenario
      config: zh-TW
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.369199731002034
    - type: f1
      value: 63.527650116059945
  - task:
      type: Clustering
    dataset:
      name: MTEB MedrxivClusteringP2P
      type: mteb/medrxiv-clustering-p2p
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 29.442504112215538
  - task:
      type: Clustering
    dataset:
      name: MTEB MedrxivClusteringS2S
      type: mteb/medrxiv-clustering-s2s
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 26.16062814161053
  - task:
      type: Retrieval
    dataset:
      name: MTEB QuoraRetrieval
      type: quora
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 65.319
    - type: map_at_10
      value: 78.72
    - type: map_at_100
      value: 79.44600000000001
    - type: map_at_1000
      value: 79.469
    - type: map_at_3
      value: 75.693
    - type: map_at_5
      value: 77.537
    - type: mrr_at_1
      value: 75.24
    - type: mrr_at_10
      value: 82.304
    - type: mrr_at_100
      value: 82.485
    - type: mrr_at_1000
      value: 82.489
    - type: mrr_at_3
      value: 81.002
    - type: mrr_at_5
      value: 81.817
    - type: ndcg_at_1
      value: 75.26
    - type: ndcg_at_10
      value: 83.07
    - type: ndcg_at_100
      value: 84.829
    - type: ndcg_at_1000
      value: 85.087
    - type: ndcg_at_3
      value: 79.67699999999999
    - type: ndcg_at_5
      value: 81.42
    - type: precision_at_1
      value: 75.26
    - type: precision_at_10
      value: 12.697
    - type: precision_at_100
      value: 1.4829999999999999
    - type: precision_at_1000
      value: 0.154
    - type: precision_at_3
      value: 34.849999999999994
    - type: precision_at_5
      value: 23.054
    - type: recall_at_1
      value: 65.319
    - type: recall_at_10
      value: 91.551
    - type: recall_at_100
      value: 98.053
    - type: recall_at_1000
      value: 99.516
    - type: recall_at_3
      value: 81.819
    - type: recall_at_5
      value: 86.66199999999999
  - task:
      type: Clustering
    dataset:
      name: MTEB RedditClustering
      type: mteb/reddit-clustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 31.249791587189996
  - task:
      type: Clustering
    dataset:
      name: MTEB RedditClusteringP2P
      type: mteb/reddit-clustering-p2p
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 43.302922383029816
  - task:
      type: STS
    dataset:
      name: MTEB SICK-R
      type: mteb/sickr-sts
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_pearson
      value: 84.80670811345861
    - type: cos_sim_spearman
      value: 79.97373018384307
    - type: euclidean_pearson
      value: 83.40205934125837
    - type: euclidean_spearman
      value: 79.73331008251854
    - type: manhattan_pearson
      value: 83.3320983393412
    - type: manhattan_spearman
      value: 79.677919746045
  - task:
      type: STS
    dataset:
      name: MTEB STS12
      type: mteb/sts12-sts
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 86.3816087627948
    - type: cos_sim_spearman
      value: 80.91314664846955
    - type: euclidean_pearson
      value: 85.10603071031096
    - type: euclidean_spearman
      value: 79.42663939501841
    - type: manhattan_pearson
      value: 85.16096376014066
    - type: manhattan_spearman
      value: 79.51936545543191
  - task:
      type: STS
    dataset:
      name: MTEB STS13
      type: mteb/sts13-sts
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 80.44665329940209
    - type: cos_sim_spearman
      value: 82.86479010707745
    - type: euclidean_pearson
      value: 84.06719627734672
    - type: euclidean_spearman
      value: 84.9356099976297
    - type: manhattan_pearson
      value: 84.10370009572624
    - type: manhattan_spearman
      value: 84.96828040546536
  - task:
      type: STS
    dataset:
      name: MTEB STS14
      type: mteb/sts14-sts
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 86.05704260568437
    - type: cos_sim_spearman
      value: 87.36399473803172
    - type: euclidean_pearson
      value: 86.8895170159388
    - type: euclidean_spearman
      value: 87.16246440866921
    - type: manhattan_pearson
      value: 86.80814774538997
    - type: manhattan_spearman
      value: 87.09320142699522
  - task:
      type: STS
    dataset:
      name: MTEB STS15
      type: mteb/sts15-sts
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 85.97825118945852
    - type: cos_sim_spearman
      value: 88.31438033558268
    - type: euclidean_pearson
      value: 87.05174694758092
    - type: euclidean_spearman
      value: 87.80659468392355
    - type: manhattan_pearson
      value: 86.98831322198717
    - type: manhattan_spearman
      value: 87.72820615049285
  - task:
      type: STS
    dataset:
      name: MTEB STS16
      type: mteb/sts16-sts
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 78.68745420126719
    - type: cos_sim_spearman
      value: 81.6058424699445
    - type: euclidean_pearson
      value: 81.16540133861879
    - type: euclidean_spearman
      value: 81.86377535458067
    - type: manhattan_pearson
      value: 81.13813317937021
    - type: manhattan_spearman
      value: 81.87079962857256
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (ko-ko)
      type: mteb/sts17-crosslingual-sts
      config: ko-ko
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 68.06192660936868
    - type: cos_sim_spearman
      value: 68.2376353514075
    - type: euclidean_pearson
      value: 60.68326946956215
    - type: euclidean_spearman
      value: 59.19352349785952
    - type: manhattan_pearson
      value: 60.6592944683418
    - type: manhattan_spearman
      value: 59.167534419270865
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (ar-ar)
      type: mteb/sts17-crosslingual-sts
      config: ar-ar
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 76.78098264855684
    - type: cos_sim_spearman
      value: 78.02670452969812
    - type: euclidean_pearson
      value: 77.26694463661255
    - type: euclidean_spearman
      value: 77.47007626009587
    - type: manhattan_pearson
      value: 77.25070088632027
    - type: manhattan_spearman
      value: 77.36368265830724
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-ar)
      type: mteb/sts17-crosslingual-sts
      config: en-ar
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 78.45418506379532
    - type: cos_sim_spearman
      value: 78.60412019902428
    - type: euclidean_pearson
      value: 79.90303710850512
    - type: euclidean_spearman
      value: 78.67123625004957
    - type: manhattan_pearson
      value: 80.09189580897753
    - type: manhattan_spearman
      value: 79.02484481441483
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-de)
      type: mteb/sts17-crosslingual-sts
      config: en-de
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 82.35556731232779
    - type: cos_sim_spearman
      value: 81.48249735354844
    - type: euclidean_pearson
      value: 81.66748026636621
    - type: euclidean_spearman
      value: 80.35571574338547
    - type: manhattan_pearson
      value: 81.38214732806365
    - type: manhattan_spearman
      value: 79.9018202958774
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-en)
      type: mteb/sts17-crosslingual-sts
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 86.4527703176897
    - type: cos_sim_spearman
      value: 85.81084095829584
    - type: euclidean_pearson
      value: 86.43489162324457
    - type: euclidean_spearman
      value: 85.27110976093296
    - type: manhattan_pearson
      value: 86.43674259444512
    - type: manhattan_spearman
      value: 85.05719308026032
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-tr)
      type: mteb/sts17-crosslingual-sts
      config: en-tr
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 76.00411240034492
    - type: cos_sim_spearman
      value: 76.33887356560854
    - type: euclidean_pearson
      value: 76.81730660019446
    - type: euclidean_spearman
      value: 75.04432185451306
    - type: manhattan_pearson
      value: 77.22298813168995
    - type: manhattan_spearman
      value: 75.56420330256725
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (es-en)
      type: mteb/sts17-crosslingual-sts
      config: es-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 79.1447136836213
    - type: cos_sim_spearman
      value: 81.80823850788917
    - type: euclidean_pearson
      value: 80.84505734814422
    - type: euclidean_spearman
      value: 81.714168092736
    - type: manhattan_pearson
      value: 80.84713816174187
    - type: manhattan_spearman
      value: 81.61267814749516
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (es-es)
      type: mteb/sts17-crosslingual-sts
      config: es-es
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 87.01257457052873
    - type: cos_sim_spearman
      value: 87.91146458004216
    - type: euclidean_pearson
      value: 88.36771859717994
    - type: euclidean_spearman
      value: 87.73182474597515
    - type: manhattan_pearson
      value: 88.26551451003671
    - type: manhattan_spearman
      value: 87.71675151388992
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (fr-en)
      type: mteb/sts17-crosslingual-sts
      config: fr-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 79.20121618382373
    - type: cos_sim_spearman
      value: 78.05794691968603
    - type: euclidean_pearson
      value: 79.93819925682054
    - type: euclidean_spearman
      value: 78.00586118701553
    - type: manhattan_pearson
      value: 80.05598625820885
    - type: manhattan_spearman
      value: 78.04802948866832
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (it-en)
      type: mteb/sts17-crosslingual-sts
      config: it-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 81.51743373871778
    - type: cos_sim_spearman
      value: 80.98266651818703
    - type: euclidean_pearson
      value: 81.11875722505269
    - type: euclidean_spearman
      value: 79.45188413284538
    - type: manhattan_pearson
      value: 80.7988457619225
    - type: manhattan_spearman
      value: 79.49643569311485
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (nl-en)
      type: mteb/sts17-crosslingual-sts
      config: nl-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 81.78679924046351
    - type: cos_sim_spearman
      value: 80.9986574147117
    - type: euclidean_pearson
      value: 82.09130079135713
    - type: euclidean_spearman
      value: 80.66215667390159
    - type: manhattan_pearson
      value: 82.0328610549654
    - type: manhattan_spearman
      value: 80.31047226932408
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (en)
      type: mteb/sts22-crosslingual-sts
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 58.08082172994642
    - type: cos_sim_spearman
      value: 62.9940530222459
    - type: euclidean_pearson
      value: 58.47927303460365
    - type: euclidean_spearman
      value: 60.8440317609258
    - type: manhattan_pearson
      value: 58.32438211697841
    - type: manhattan_spearman
      value: 60.69642636776064
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de)
      type: mteb/sts22-crosslingual-sts
      config: de
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 33.83985707464123
    - type: cos_sim_spearman
      value: 46.89093209603036
    - type: euclidean_pearson
      value: 34.63602187576556
    - type: euclidean_spearman
      value: 46.31087228200712
    - type: manhattan_pearson
      value: 34.66899391543166
    - type: manhattan_spearman
      value: 46.33049538425276
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (es)
      type: mteb/sts22-crosslingual-sts
      config: es
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 51.61315965767736
    - type: cos_sim_spearman
      value: 58.9434266730386
    - type: euclidean_pearson
      value: 50.35885602217862
    - type: euclidean_spearman
      value: 58.238679883286025
    - type: manhattan_pearson
      value: 53.01732044381151
    - type: manhattan_spearman
      value: 58.10482351761412
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (pl)
      type: mteb/sts22-crosslingual-sts
      config: pl
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 26.771738440430177
    - type: cos_sim_spearman
      value: 34.807259227816054
    - type: euclidean_pearson
      value: 17.82657835823811
    - type: euclidean_spearman
      value: 34.27912898498941
    - type: manhattan_pearson
      value: 19.121527758886312
    - type: manhattan_spearman
      value: 34.4940050226265
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (tr)
      type: mteb/sts22-crosslingual-sts
      config: tr
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 52.8354704676683
    - type: cos_sim_spearman
      value: 57.28629534815841
    - type: euclidean_pearson
      value: 54.10329332004385
    - type: euclidean_spearman
      value: 58.15030615859976
    - type: manhattan_pearson
      value: 55.42372087433115
    - type: manhattan_spearman
      value: 57.52270736584036
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (ar)
      type: mteb/sts22-crosslingual-sts
      config: ar
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 31.01976557986924
    - type: cos_sim_spearman
      value: 54.506959483927616
    - type: euclidean_pearson
      value: 36.917863022119086
    - type: euclidean_spearman
      value: 53.750194241538566
    - type: manhattan_pearson
      value: 37.200177833241085
    - type: manhattan_spearman
      value: 53.507659188082535
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (ru)
      type: mteb/sts22-crosslingual-sts
      config: ru
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 46.38635647225934
    - type: cos_sim_spearman
      value: 54.50892732637536
    - type: euclidean_pearson
      value: 40.8331015184763
    - type: euclidean_spearman
      value: 53.142903182230924
    - type: manhattan_pearson
      value: 43.07655692906317
    - type: manhattan_spearman
      value: 53.5833474125901
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (zh)
      type: mteb/sts22-crosslingual-sts
      config: zh
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 60.52525456662916
    - type: cos_sim_spearman
      value: 63.23975489531082
    - type: euclidean_pearson
      value: 58.989191722317514
    - type: euclidean_spearman
      value: 62.536326639863894
    - type: manhattan_pearson
      value: 61.32982866201855
    - type: manhattan_spearman
      value: 63.068262822520516
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (fr)
      type: mteb/sts22-crosslingual-sts
      config: fr
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 59.63798684577696
    - type: cos_sim_spearman
      value: 74.09937723367189
    - type: euclidean_pearson
      value: 63.77494904383906
    - type: euclidean_spearman
      value: 71.15932571292481
    - type: manhattan_pearson
      value: 63.69646122775205
    - type: manhattan_spearman
      value: 70.54960698541632
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de-en)
      type: mteb/sts22-crosslingual-sts
      config: de-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 36.50262468726711
    - type: cos_sim_spearman
      value: 45.00322499674274
    - type: euclidean_pearson
      value: 32.58759216581778
    - type: euclidean_spearman
      value: 40.13720951315429
    - type: manhattan_pearson
      value: 34.88422299605277
    - type: manhattan_spearman
      value: 40.63516862200963
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (es-en)
      type: mteb/sts22-crosslingual-sts
      config: es-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 56.498552617040275
    - type: cos_sim_spearman
      value: 67.71358426124443
    - type: euclidean_pearson
      value: 57.16474781778287
    - type: euclidean_spearman
      value: 65.721515493531
    - type: manhattan_pearson
      value: 59.25227610738926
    - type: manhattan_spearman
      value: 65.89743680340739
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (it)
      type: mteb/sts22-crosslingual-sts
      config: it
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 55.97978814727984
    - type: cos_sim_spearman
      value: 65.85821395092104
    - type: euclidean_pearson
      value: 59.11117270978519
    - type: euclidean_spearman
      value: 64.50062069934965
    - type: manhattan_pearson
      value: 59.4436213778161
    - type: manhattan_spearman
      value: 64.4003273074382
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (pl-en)
      type: mteb/sts22-crosslingual-sts
      config: pl-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 58.00873192515712
    - type: cos_sim_spearman
      value: 60.167708809138745
    - type: euclidean_pearson
      value: 56.91950637760252
    - type: euclidean_spearman
      value: 58.50593399441014
    - type: manhattan_pearson
      value: 58.683747352584994
    - type: manhattan_spearman
      value: 59.38110066799761
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (zh-en)
      type: mteb/sts22-crosslingual-sts
      config: zh-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 54.26020658151187
    - type: cos_sim_spearman
      value: 61.29236187204147
    - type: euclidean_pearson
      value: 55.993896804147056
    - type: euclidean_spearman
      value: 58.654928232615354
    - type: manhattan_pearson
      value: 56.612492816099426
    - type: manhattan_spearman
      value: 58.65144067094258
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (es-it)
      type: mteb/sts22-crosslingual-sts
      config: es-it
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 49.13817835368122
    - type: cos_sim_spearman
      value: 50.78524216975442
    - type: euclidean_pearson
      value: 46.56046454501862
    - type: euclidean_spearman
      value: 50.3935060082369
    - type: manhattan_pearson
      value: 48.0232348418531
    - type: manhattan_spearman
      value: 50.79528358464199
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de-fr)
      type: mteb/sts22-crosslingual-sts
      config: de-fr
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 44.274388638585286
    - type: cos_sim_spearman
      value: 49.43124017389838
    - type: euclidean_pearson
      value: 42.45909582681174
    - type: euclidean_spearman
      value: 49.661383797129055
    - type: manhattan_pearson
      value: 42.5771970142383
    - type: manhattan_spearman
      value: 50.14423414390715
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de-pl)
      type: mteb/sts22-crosslingual-sts
      config: de-pl
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 26.119500839749776
    - type: cos_sim_spearman
      value: 39.324070169024424
    - type: euclidean_pearson
      value: 35.83247077201831
    - type: euclidean_spearman
      value: 42.61903924348457
    - type: manhattan_pearson
      value: 35.50415034487894
    - type: manhattan_spearman
      value: 41.87998075949351
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (fr-pl)
      type: mteb/sts22-crosslingual-sts
      config: fr-pl
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 72.62575835691209
    - type: cos_sim_spearman
      value: 73.24670207647144
    - type: euclidean_pearson
      value: 78.07793323914657
    - type: euclidean_spearman
      value: 73.24670207647144
    - type: manhattan_pearson
      value: 77.51429306378206
    - type: manhattan_spearman
      value: 73.24670207647144
  - task:
      type: STS
    dataset:
      name: MTEB STSBenchmark
      type: mteb/stsbenchmark-sts
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 84.09375596849891
    - type: cos_sim_spearman
      value: 86.44881302053585
    - type: euclidean_pearson
      value: 84.71259163967213
    - type: euclidean_spearman
      value: 85.63661992344069
    - type: manhattan_pearson
      value: 84.64466537502614
    - type: manhattan_spearman
      value: 85.53769949940238
  - task:
      type: Reranking
    dataset:
      name: MTEB SciDocsRR
      type: mteb/scidocs-reranking
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 70.2056154684549
    - type: mrr
      value: 89.52703161036494
  - task:
      type: PairClassification
    dataset:
      name: MTEB SprintDuplicateQuestions
      type: mteb/sprintduplicatequestions-pairclassification
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.57623762376238
    - type: cos_sim_ap
      value: 83.53051588811371
    - type: cos_sim_f1
      value: 77.72704211060375
    - type: cos_sim_precision
      value: 78.88774459320288
    - type: cos_sim_recall
      value: 76.6
    - type: dot_accuracy
      value: 99.06435643564356
    - type: dot_ap
      value: 27.003124923857463
    - type: dot_f1
      value: 34.125269978401725
    - type: dot_precision
      value: 37.08920187793427
    - type: dot_recall
      value: 31.6
    - type: euclidean_accuracy
      value: 99.61485148514852
    - type: euclidean_ap
      value: 85.47332647001774
    - type: euclidean_f1
      value: 80.0808897876643
    - type: euclidean_precision
      value: 80.98159509202453
    - type: euclidean_recall
      value: 79.2
    - type: manhattan_accuracy
      value: 99.61683168316831
    - type: manhattan_ap
      value: 85.41969859598552
    - type: manhattan_f1
      value: 79.77755308392315
    - type: manhattan_precision
      value: 80.67484662576688
    - type: manhattan_recall
      value: 78.9
    - type: max_accuracy
      value: 99.61683168316831
    - type: max_ap
      value: 85.47332647001774
    - type: max_f1
      value: 80.0808897876643
  - task:
      type: Clustering
    dataset:
      name: MTEB StackExchangeClustering
      type: mteb/stackexchange-clustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 34.35688940053467
  - task:
      type: Clustering
    dataset:
      name: MTEB StackExchangeClusteringP2P
      type: mteb/stackexchange-clustering-p2p
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 30.64427069276576
  - task:
      type: Reranking
    dataset:
      name: MTEB StackOverflowDupQuestions
      type: mteb/stackoverflowdupquestions-reranking
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 44.89500754900078
    - type: mrr
      value: 45.33215558950853
  - task:
      type: Summarization
    dataset:
      name: MTEB SummEval
      type: mteb/summeval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.653069624224084
    - type: cos_sim_spearman
      value: 30.10187112430319
    - type: dot_pearson
      value: 28.966278202103666
    - type: dot_spearman
      value: 28.342234095507767
  - task:
      type: Classification
    dataset:
      name: MTEB ToxicConversationsClassification
      type: mteb/toxic_conversations_50k
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 65.96839999999999
    - type: ap
      value: 11.846327590186444
    - type: f1
      value: 50.518102944693574
  - task:
      type: Classification
    dataset:
      name: MTEB TweetSentimentExtractionClassification
      type: mteb/tweet_sentiment_extraction
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 55.220713073005086
    - type: f1
      value: 55.47856175692088
  - task:
      type: Clustering
    dataset:
      name: MTEB TwentyNewsgroupsClustering
      type: mteb/twentynewsgroups-clustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 31.581473892235877
  - task:
      type: PairClassification
    dataset:
      name: MTEB TwitterSemEval2015
      type: mteb/twittersemeval2015-pairclassification
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 82.94093103653812
    - type: cos_sim_ap
      value: 62.48963249213361
    - type: cos_sim_f1
      value: 58.9541137429912
    - type: cos_sim_precision
      value: 52.05091937765205
    - type: cos_sim_recall
      value: 67.96833773087072
    - type: dot_accuracy
      value: 78.24998509864696
    - type: dot_ap
      value: 40.82371294480071
    - type: dot_f1
      value: 44.711163153786096
    - type: dot_precision
      value: 35.475379374419326
    - type: dot_recall
      value: 60.4485488126649
    - type: euclidean_accuracy
      value: 83.13166835548668
    - type: euclidean_ap
      value: 63.459878609769774
    - type: euclidean_f1
      value: 60.337199569532466
    - type: euclidean_precision
      value: 55.171659741963694
    - type: euclidean_recall
      value: 66.56992084432719
    - type: manhattan_accuracy
      value: 83.00649698992669
    - type: manhattan_ap
      value: 63.263161177904905
    - type: manhattan_f1
      value: 60.17122874713614
    - type: manhattan_precision
      value: 55.40750610703975
    - type: manhattan_recall
      value: 65.8311345646438
    - type: max_accuracy
      value: 83.13166835548668
    - type: max_ap
      value: 63.459878609769774
    - type: max_f1
      value: 60.337199569532466
  - task:
      type: PairClassification
    dataset:
      name: MTEB TwitterURLCorpus
      type: mteb/twitterurlcorpus-pairclassification
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 87.80416812201653
    - type: cos_sim_ap
      value: 83.45540469219863
    - type: cos_sim_f1
      value: 75.58836427422892
    - type: cos_sim_precision
      value: 71.93934335002783
    - type: cos_sim_recall
      value: 79.62734832152756
    - type: dot_accuracy
      value: 83.04226336011176
    - type: dot_ap
      value: 70.63007268018524
    - type: dot_f1
      value: 65.35980325765405
    - type: dot_precision
      value: 60.84677151768532
    - type: dot_recall
      value: 70.59593470896212
    - type: euclidean_accuracy
      value: 87.60430007373773
    - type: euclidean_ap
      value: 83.10068502536592
    - type: euclidean_f1
      value: 75.02510506936439
    - type: euclidean_precision
      value: 72.56637168141593
    - type: euclidean_recall
      value: 77.65629812134279
    - type: manhattan_accuracy
      value: 87.60041914076145
    - type: manhattan_ap
      value: 83.05480769911229
    - type: manhattan_f1
      value: 74.98522895125554
    - type: manhattan_precision
      value: 72.04797047970479
    - type: manhattan_recall
      value: 78.17215891592238
    - type: max_accuracy
      value: 87.80416812201653
    - type: max_ap
      value: 83.45540469219863
    - type: max_f1
      value: 75.58836427422892
---
# shibing624/text2vec-base-multilingual
This is a CoSENT(Cosine Sentence) model: shibing624/text2vec-base-multilingual.

It maps sentences to a 384 dimensional dense vector space and can be used for tasks 
like sentence embeddings, text matching or semantic search.



- training dataset: https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-multilingual-dataset
- base model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- max_seq_length: 256
- best epoch: 4
- sentence embedding dim: 384

## Evaluation
For an automated evaluation of this model, see the *Evaluation Benchmark*: [text2vec](https://github.com/shibing624/text2vec)
## Languages
Available languages are: de, en, es, fr, it, nl, pl, pt, ru, zh

### Release Models

- 本项目release模型的中文匹配评测结果：

| Arch       | BaseModel                                                    | Model                                                                                                                                             | ATEC  |  BQ   | LCQMC | PAWSX | STS-B | SOHU-dd | SOHU-dc |    Avg    |  QPS  |
|:-----------|:-------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:---------:|:-----:|
| Word2Vec   | word2vec                                                     | [w2v-light-tencent-chinese](https://ai.tencent.com/ailab/nlp/en/download.html)                                                                    | 20.00 | 31.49 | 59.46 | 2.57  | 55.78 |  55.04  |  20.70  |   35.03   | 23769 |
| SBERT      | xlm-roberta-base                                             | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 18.42 | 38.52 | 63.96 | 10.14 | 78.90 |  63.01  |  52.28  |   46.46   | 3138  |
| Instructor | hfl/chinese-roberta-wwm-ext                                  | [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)                                                                                       | 41.27 | 63.81 | 74.87 | 12.20 | 76.96 |  75.83  |  60.55  |   57.93   | 2980  |
| CoSENT     | hfl/chinese-macbert-base                                     | [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)                                                       | 31.93 | 42.67 | 70.16 | 17.21 | 79.30 |  70.27  |  50.42  |   51.61   | 3008  |
| CoSENT     | hfl/chinese-lert-large                                       | [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)                                                   | 32.61 | 44.59 | 69.30 | 14.51 | 79.44 |  73.01  |  59.04  |   53.12   | 2092  |
| CoSENT     | nghuyong/ernie-3.0-base-zh                                   | [shibing624/text2vec-base-chinese-sentence](https://huggingface.co/shibing624/text2vec-base-chinese-sentence)                                     | 43.37 | 61.43 | 73.48 | 38.90 | 78.25 |  70.60  |  53.08  |   59.87   | 3089  |
| CoSENT     | nghuyong/ernie-3.0-base-zh                                   | [shibing624/text2vec-base-chinese-paraphrase](https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase)                                 | 44.89 | 63.58 | 74.24 | 40.90 | 78.93 |  76.70  |  63.30  | **63.08** | 3066  |
| CoSENT     | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  | [shibing624/text2vec-base-multilingual](https://huggingface.co/shibing624/text2vec-base-multilingual)                                             | 32.39 | 50.33 | 65.64 | 32.56 | 74.45 |  68.88  |  51.17  |   53.67   | 4004  |


说明：
- 结果评测指标：spearman系数
- `shibing624/text2vec-base-chinese`模型，是用CoSENT方法训练，基于`hfl/chinese-macbert-base`在中文STS-B数据训练得到，并在中文STS-B测试集评估达到较好效果，运行[examples/training_sup_text_matching_model.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model.py)代码可训练模型，模型文件已经上传HF model hub，中文通用语义匹配任务推荐使用
- `shibing624/text2vec-base-chinese-sentence`模型，是用CoSENT方法训练，基于`nghuyong/ernie-3.0-base-zh`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-sentence-dataset)训练得到，并在中文各NLI测试集评估达到较好效果，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2s(句子vs句子)语义匹配任务推荐使用
- `shibing624/text2vec-base-chinese-paraphrase`模型，是用CoSENT方法训练，基于`nghuyong/ernie-3.0-base-zh`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-paraphrase-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-paraphrase-dataset)，数据集相对于[shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-sentence-dataset)加入了s2p(sentence to paraphrase)数据，强化了其长文本的表征能力，并在中文各NLI测试集评估达到SOTA，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2p(句子vs段落)语义匹配任务推荐使用
- `shibing624/text2vec-base-multilingual`模型，是用CoSENT方法训练，基于`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`用人工挑选后的多语言STS数据集[shibing624/nli-zh-all/text2vec-base-multilingual-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-multilingual-dataset)训练得到，并在中英文测试集评估相对于原模型效果有提升，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，多语言语义匹配任务推荐使用
- `w2v-light-tencent-chinese`是腾讯词向量的Word2Vec模型，CPU加载使用，适用于中文字面匹配任务和缺少数据的冷启动情况
- QPS的GPU测试环境是Tesla V100，显存32GB

模型训练实验报告：[实验报告](https://github.com/shibing624/text2vec/blob/master/docs/model_report.md)

## Usage (text2vec)
Using this model becomes easy when you have [text2vec](https://github.com/shibing624/text2vec) installed:

```
pip install -U text2vec
```

Then you can use the model like this:

```python
from text2vec import SentenceModel
sentences = ['如何更换花呗绑定银行卡', 'How to replace the Huabei bundled bank card']

model = SentenceModel('shibing624/text2vec-base-multilingual')
embeddings = model.encode(sentences)
print(embeddings)
```

## Usage (HuggingFace Transformers)
Without [text2vec](https://github.com/shibing624/text2vec), you can use the model like this: 

First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

Install transformers:
```
pip install transformers
```

Then load model and predict:
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('shibing624/text2vec-base-multilingual')
model = AutoModel.from_pretrained('shibing624/text2vec-base-multilingual')
sentences = ['如何更换花呗绑定银行卡', 'How to replace the Huabei bundled bank card']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings)
```

## Usage (sentence-transformers)
[sentence-transformers](https://github.com/UKPLab/sentence-transformers) is a popular library to compute dense vector representations for sentences.

Install sentence-transformers:
```
pip install -U sentence-transformers
```

Then load model and predict:

```python
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("shibing624/text2vec-base-multilingual")
sentences = ['如何更换花呗绑定银行卡', 'How to replace the Huabei bundled bank card']

sentence_embeddings = m.encode(sentences)
print("Sentence embeddings:")
print(sentence_embeddings)
```


## Full Model Architecture
```
CoSENT(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_mean_tokens': True})
)
```


## Intended uses

Our model is intented to be used as a sentence and short paragraph encoder. Given an input text, it ouptuts a vector which captures 
the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

By default, input text longer than 256 word pieces is truncated.


## Training procedure

### Pre-training 

We use the pretrained [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) model. 
Please refer to the model card for more detailed information about the pre-training procedure.

### Fine-tuning 

We fine-tune the model using a contrastive objective. Formally, we compute the cosine similarity from each 
possible sentence pairs from the batch.
We then apply the rank loss by comparing with true pairs and false pairs.


## Citing & Authors
This model was trained by [text2vec](https://github.com/shibing624/text2vec). 
        
If you find this model helpful, feel free to cite:
```bibtex 
@software{text2vec,
  author = {Ming Xu},
  title = {text2vec: A Tool for Text to Vector},
  year = {2023},
  url = {https://github.com/shibing624/text2vec},
}
```