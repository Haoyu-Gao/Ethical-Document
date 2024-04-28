---
language:
- multilingual
- af
- am
- ar
- as
- az
- be
- bg
- bn
- br
- bs
- ca
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- he
- hi
- hr
- hu
- hy
- id
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lo
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- my
- ne
- nl
- 'no'
- om
- or
- pa
- pl
- ps
- pt
- ro
- ru
- sa
- sd
- si
- sk
- sl
- so
- sq
- sr
- su
- sv
- sw
- ta
- te
- th
- tl
- tr
- ug
- uk
- ur
- uz
- vi
- xh
- yi
- zh
license: mit
tags:
- mteb
- Sentence Transformers
- sentence-similarity
- sentence-transformers
model-index:
- name: multilingual-e5-small
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
      value: 73.79104477611939
    - type: ap
      value: 36.9996434842022
    - type: f1
      value: 67.95453679103099
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
      value: 71.64882226980728
    - type: ap
      value: 82.11942130026586
    - type: f1
      value: 69.87963421606715
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
      value: 75.8095952023988
    - type: ap
      value: 24.46869495579561
    - type: f1
      value: 63.00108480037597
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
      value: 64.186295503212
    - type: ap
      value: 15.496804690197042
    - type: f1
      value: 52.07153895475031
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
      value: 88.699325
    - type: ap
      value: 85.27039559917269
    - type: f1
      value: 88.65556295032513
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
      value: 44.69799999999999
    - type: f1
      value: 43.73187348654165
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
      value: 40.245999999999995
    - type: f1
      value: 39.3863530637684
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
      value: 40.394
    - type: f1
      value: 39.301223469483446
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
      value: 38.864
    - type: f1
      value: 37.97974261868003
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
      value: 37.682
    - type: f1
      value: 37.07399369768313
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
      value: 37.504
    - type: f1
      value: 36.62317273874278
  - task:
      type: Retrieval
    dataset:
      name: MTEB ArguAna
      type: arguana
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 19.061
    - type: map_at_10
      value: 31.703
    - type: map_at_100
      value: 32.967
    - type: map_at_1000
      value: 33.001000000000005
    - type: map_at_3
      value: 27.466
    - type: map_at_5
      value: 29.564
    - type: mrr_at_1
      value: 19.559
    - type: mrr_at_10
      value: 31.874999999999996
    - type: mrr_at_100
      value: 33.146
    - type: mrr_at_1000
      value: 33.18
    - type: mrr_at_3
      value: 27.667
    - type: mrr_at_5
      value: 29.74
    - type: ndcg_at_1
      value: 19.061
    - type: ndcg_at_10
      value: 39.062999999999995
    - type: ndcg_at_100
      value: 45.184000000000005
    - type: ndcg_at_1000
      value: 46.115
    - type: ndcg_at_3
      value: 30.203000000000003
    - type: ndcg_at_5
      value: 33.953
    - type: precision_at_1
      value: 19.061
    - type: precision_at_10
      value: 6.279999999999999
    - type: precision_at_100
      value: 0.9129999999999999
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 12.706999999999999
    - type: precision_at_5
      value: 9.431000000000001
    - type: recall_at_1
      value: 19.061
    - type: recall_at_10
      value: 62.802
    - type: recall_at_100
      value: 91.323
    - type: recall_at_1000
      value: 98.72
    - type: recall_at_3
      value: 38.122
    - type: recall_at_5
      value: 47.155
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
      value: 39.22266660528253
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
      value: 30.79980849482483
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
      value: 57.8790068352054
    - type: mrr
      value: 71.78791276436706
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
      value: 82.36328364043163
    - type: cos_sim_spearman
      value: 82.26211536195868
    - type: euclidean_pearson
      value: 80.3183865039173
    - type: euclidean_spearman
      value: 79.88495276296132
    - type: manhattan_pearson
      value: 80.14484480692127
    - type: manhattan_spearman
      value: 80.39279565980743
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (de-en)
      type: mteb/bucc-bitext-mining
      config: de-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 98.0375782881002
    - type: f1
      value: 97.86012526096033
    - type: precision
      value: 97.77139874739039
    - type: recall
      value: 98.0375782881002
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (fr-en)
      type: mteb/bucc-bitext-mining
      config: fr-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 93.35241030156286
    - type: f1
      value: 92.66050333846944
    - type: precision
      value: 92.3306919069631
    - type: recall
      value: 93.35241030156286
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (ru-en)
      type: mteb/bucc-bitext-mining
      config: ru-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 94.0699688257707
    - type: f1
      value: 93.50236693222492
    - type: precision
      value: 93.22791825424315
    - type: recall
      value: 94.0699688257707
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (zh-en)
      type: mteb/bucc-bitext-mining
      config: zh-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 89.25750394944708
    - type: f1
      value: 88.79234684921889
    - type: precision
      value: 88.57293312269616
    - type: recall
      value: 89.25750394944708
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
      value: 79.41558441558442
    - type: f1
      value: 79.25886487487219
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
      value: 35.747820820329736
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
      value: 27.045143830596146
  - task:
      type: Retrieval
    dataset:
      name: MTEB CQADupstackRetrieval
      type: BeIR/cqadupstack
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.252999999999997
    - type: map_at_10
      value: 31.655916666666666
    - type: map_at_100
      value: 32.680749999999996
    - type: map_at_1000
      value: 32.79483333333334
    - type: map_at_3
      value: 29.43691666666666
    - type: map_at_5
      value: 30.717416666666665
    - type: mrr_at_1
      value: 28.602750000000004
    - type: mrr_at_10
      value: 35.56875
    - type: mrr_at_100
      value: 36.3595
    - type: mrr_at_1000
      value: 36.427749999999996
    - type: mrr_at_3
      value: 33.586166666666664
    - type: mrr_at_5
      value: 34.73641666666666
    - type: ndcg_at_1
      value: 28.602750000000004
    - type: ndcg_at_10
      value: 36.06933333333334
    - type: ndcg_at_100
      value: 40.70141666666667
    - type: ndcg_at_1000
      value: 43.24341666666667
    - type: ndcg_at_3
      value: 32.307916666666664
    - type: ndcg_at_5
      value: 34.129999999999995
    - type: precision_at_1
      value: 28.602750000000004
    - type: precision_at_10
      value: 6.097666666666667
    - type: precision_at_100
      value: 0.9809166666666668
    - type: precision_at_1000
      value: 0.13766666666666663
    - type: precision_at_3
      value: 14.628166666666667
    - type: precision_at_5
      value: 10.266916666666667
    - type: recall_at_1
      value: 24.252999999999997
    - type: recall_at_10
      value: 45.31916666666667
    - type: recall_at_100
      value: 66.03575000000001
    - type: recall_at_1000
      value: 83.94708333333334
    - type: recall_at_3
      value: 34.71941666666666
    - type: recall_at_5
      value: 39.46358333333333
  - task:
      type: Retrieval
    dataset:
      name: MTEB ClimateFEVER
      type: climate-fever
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 9.024000000000001
    - type: map_at_10
      value: 15.644
    - type: map_at_100
      value: 17.154
    - type: map_at_1000
      value: 17.345
    - type: map_at_3
      value: 13.028
    - type: map_at_5
      value: 14.251
    - type: mrr_at_1
      value: 19.674
    - type: mrr_at_10
      value: 29.826999999999998
    - type: mrr_at_100
      value: 30.935000000000002
    - type: mrr_at_1000
      value: 30.987
    - type: mrr_at_3
      value: 26.645000000000003
    - type: mrr_at_5
      value: 28.29
    - type: ndcg_at_1
      value: 19.674
    - type: ndcg_at_10
      value: 22.545
    - type: ndcg_at_100
      value: 29.207
    - type: ndcg_at_1000
      value: 32.912
    - type: ndcg_at_3
      value: 17.952
    - type: ndcg_at_5
      value: 19.363
    - type: precision_at_1
      value: 19.674
    - type: precision_at_10
      value: 7.212000000000001
    - type: precision_at_100
      value: 1.435
    - type: precision_at_1000
      value: 0.212
    - type: precision_at_3
      value: 13.507
    - type: precision_at_5
      value: 10.397
    - type: recall_at_1
      value: 9.024000000000001
    - type: recall_at_10
      value: 28.077999999999996
    - type: recall_at_100
      value: 51.403
    - type: recall_at_1000
      value: 72.406
    - type: recall_at_3
      value: 16.768
    - type: recall_at_5
      value: 20.737
  - task:
      type: Retrieval
    dataset:
      name: MTEB DBPedia
      type: dbpedia-entity
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 8.012
    - type: map_at_10
      value: 17.138
    - type: map_at_100
      value: 24.146
    - type: map_at_1000
      value: 25.622
    - type: map_at_3
      value: 12.552
    - type: map_at_5
      value: 14.435
    - type: mrr_at_1
      value: 62.25000000000001
    - type: mrr_at_10
      value: 71.186
    - type: mrr_at_100
      value: 71.504
    - type: mrr_at_1000
      value: 71.514
    - type: mrr_at_3
      value: 69.333
    - type: mrr_at_5
      value: 70.408
    - type: ndcg_at_1
      value: 49.75
    - type: ndcg_at_10
      value: 37.76
    - type: ndcg_at_100
      value: 42.071
    - type: ndcg_at_1000
      value: 49.309
    - type: ndcg_at_3
      value: 41.644
    - type: ndcg_at_5
      value: 39.812999999999995
    - type: precision_at_1
      value: 62.25000000000001
    - type: precision_at_10
      value: 30.15
    - type: precision_at_100
      value: 9.753
    - type: precision_at_1000
      value: 1.9189999999999998
    - type: precision_at_3
      value: 45.667
    - type: precision_at_5
      value: 39.15
    - type: recall_at_1
      value: 8.012
    - type: recall_at_10
      value: 22.599
    - type: recall_at_100
      value: 48.068
    - type: recall_at_1000
      value: 71.328
    - type: recall_at_3
      value: 14.043
    - type: recall_at_5
      value: 17.124
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
      value: 42.455
    - type: f1
      value: 37.59462649781862
  - task:
      type: Retrieval
    dataset:
      name: MTEB FEVER
      type: fever
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 58.092
    - type: map_at_10
      value: 69.586
    - type: map_at_100
      value: 69.968
    - type: map_at_1000
      value: 69.982
    - type: map_at_3
      value: 67.48100000000001
    - type: map_at_5
      value: 68.915
    - type: mrr_at_1
      value: 62.166
    - type: mrr_at_10
      value: 73.588
    - type: mrr_at_100
      value: 73.86399999999999
    - type: mrr_at_1000
      value: 73.868
    - type: mrr_at_3
      value: 71.6
    - type: mrr_at_5
      value: 72.99
    - type: ndcg_at_1
      value: 62.166
    - type: ndcg_at_10
      value: 75.27199999999999
    - type: ndcg_at_100
      value: 76.816
    - type: ndcg_at_1000
      value: 77.09700000000001
    - type: ndcg_at_3
      value: 71.36
    - type: ndcg_at_5
      value: 73.785
    - type: precision_at_1
      value: 62.166
    - type: precision_at_10
      value: 9.716
    - type: precision_at_100
      value: 1.065
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 28.278
    - type: precision_at_5
      value: 18.343999999999998
    - type: recall_at_1
      value: 58.092
    - type: recall_at_10
      value: 88.73400000000001
    - type: recall_at_100
      value: 95.195
    - type: recall_at_1000
      value: 97.04599999999999
    - type: recall_at_3
      value: 78.45
    - type: recall_at_5
      value: 84.316
  - task:
      type: Retrieval
    dataset:
      name: MTEB FiQA2018
      type: fiqa
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 16.649
    - type: map_at_10
      value: 26.457000000000004
    - type: map_at_100
      value: 28.169
    - type: map_at_1000
      value: 28.352
    - type: map_at_3
      value: 23.305
    - type: map_at_5
      value: 25.169000000000004
    - type: mrr_at_1
      value: 32.407000000000004
    - type: mrr_at_10
      value: 40.922
    - type: mrr_at_100
      value: 41.931000000000004
    - type: mrr_at_1000
      value: 41.983
    - type: mrr_at_3
      value: 38.786
    - type: mrr_at_5
      value: 40.205999999999996
    - type: ndcg_at_1
      value: 32.407000000000004
    - type: ndcg_at_10
      value: 33.314
    - type: ndcg_at_100
      value: 40.312
    - type: ndcg_at_1000
      value: 43.685
    - type: ndcg_at_3
      value: 30.391000000000002
    - type: ndcg_at_5
      value: 31.525
    - type: precision_at_1
      value: 32.407000000000004
    - type: precision_at_10
      value: 8.966000000000001
    - type: precision_at_100
      value: 1.6019999999999999
    - type: precision_at_1000
      value: 0.22200000000000003
    - type: precision_at_3
      value: 20.165
    - type: precision_at_5
      value: 14.722
    - type: recall_at_1
      value: 16.649
    - type: recall_at_10
      value: 39.117000000000004
    - type: recall_at_100
      value: 65.726
    - type: recall_at_1000
      value: 85.784
    - type: recall_at_3
      value: 27.914
    - type: recall_at_5
      value: 33.289
  - task:
      type: Retrieval
    dataset:
      name: MTEB HotpotQA
      type: hotpotqa
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 36.253
    - type: map_at_10
      value: 56.16799999999999
    - type: map_at_100
      value: 57.06099999999999
    - type: map_at_1000
      value: 57.126
    - type: map_at_3
      value: 52.644999999999996
    - type: map_at_5
      value: 54.909
    - type: mrr_at_1
      value: 72.505
    - type: mrr_at_10
      value: 79.66
    - type: mrr_at_100
      value: 79.869
    - type: mrr_at_1000
      value: 79.88
    - type: mrr_at_3
      value: 78.411
    - type: mrr_at_5
      value: 79.19800000000001
    - type: ndcg_at_1
      value: 72.505
    - type: ndcg_at_10
      value: 65.094
    - type: ndcg_at_100
      value: 68.219
    - type: ndcg_at_1000
      value: 69.515
    - type: ndcg_at_3
      value: 59.99
    - type: ndcg_at_5
      value: 62.909000000000006
    - type: precision_at_1
      value: 72.505
    - type: precision_at_10
      value: 13.749
    - type: precision_at_100
      value: 1.619
    - type: precision_at_1000
      value: 0.179
    - type: precision_at_3
      value: 38.357
    - type: precision_at_5
      value: 25.313000000000002
    - type: recall_at_1
      value: 36.253
    - type: recall_at_10
      value: 68.744
    - type: recall_at_100
      value: 80.925
    - type: recall_at_1000
      value: 89.534
    - type: recall_at_3
      value: 57.535000000000004
    - type: recall_at_5
      value: 63.282000000000004
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
      value: 80.82239999999999
    - type: ap
      value: 75.65895781725314
    - type: f1
      value: 80.75880969095746
  - task:
      type: Retrieval
    dataset:
      name: MTEB MSMARCO
      type: msmarco
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 21.624
    - type: map_at_10
      value: 34.075
    - type: map_at_100
      value: 35.229
    - type: map_at_1000
      value: 35.276999999999994
    - type: map_at_3
      value: 30.245
    - type: map_at_5
      value: 32.42
    - type: mrr_at_1
      value: 22.264
    - type: mrr_at_10
      value: 34.638000000000005
    - type: mrr_at_100
      value: 35.744
    - type: mrr_at_1000
      value: 35.787
    - type: mrr_at_3
      value: 30.891000000000002
    - type: mrr_at_5
      value: 33.042
    - type: ndcg_at_1
      value: 22.264
    - type: ndcg_at_10
      value: 40.991
    - type: ndcg_at_100
      value: 46.563
    - type: ndcg_at_1000
      value: 47.743
    - type: ndcg_at_3
      value: 33.198
    - type: ndcg_at_5
      value: 37.069
    - type: precision_at_1
      value: 22.264
    - type: precision_at_10
      value: 6.5089999999999995
    - type: precision_at_100
      value: 0.9299999999999999
    - type: precision_at_1000
      value: 0.10300000000000001
    - type: precision_at_3
      value: 14.216999999999999
    - type: precision_at_5
      value: 10.487
    - type: recall_at_1
      value: 21.624
    - type: recall_at_10
      value: 62.303
    - type: recall_at_100
      value: 88.124
    - type: recall_at_1000
      value: 97.08
    - type: recall_at_3
      value: 41.099999999999994
    - type: recall_at_5
      value: 50.381
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
      value: 91.06703146374831
    - type: f1
      value: 90.86867815863172
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
      value: 87.46970977740209
    - type: f1
      value: 86.36832872036588
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
      value: 89.26951300867245
    - type: f1
      value: 88.93561193959502
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
      value: 84.22799874725963
    - type: f1
      value: 84.30490069236556
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
      value: 86.02007888131948
    - type: f1
      value: 85.39376041027991
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
      value: 85.34900542495481
    - type: f1
      value: 85.39859673336713
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
      value: 71.078431372549
    - type: f1
      value: 53.45071102002276
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
      value: 65.85798816568047
    - type: f1
      value: 46.53112748993529
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
      value: 67.96864576384256
    - type: f1
      value: 45.966703022829506
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
      value: 61.31537738803633
    - type: f1
      value: 45.52601712835461
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
      value: 66.29616349946218
    - type: f1
      value: 47.24166485726613
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
      value: 67.51537070524412
    - type: f1
      value: 49.463476319014276
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
      value: 57.06792199058508
    - type: f1
      value: 54.094921857502285
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
      value: 51.960322797579025
    - type: f1
      value: 48.547371223370945
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
      value: 54.425016812373904
    - type: f1
      value: 50.47069202054312
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
      value: 59.798251513113655
    - type: f1
      value: 57.05013069086648
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
      value: 59.37794216543376
    - type: f1
      value: 56.3607992649805
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
      value: 46.56018829858777
    - type: f1
      value: 43.87319715715134
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
      value: 62.9724277067922
    - type: f1
      value: 59.36480066245562
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
      value: 62.72696704774715
    - type: f1
      value: 59.143595966615855
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
      value: 61.5971755211836
    - type: f1
      value: 59.169445724946726
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
      value: 70.29589778076665
    - type: f1
      value: 67.7577001808977
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
      value: 66.31136516476126
    - type: f1
      value: 64.52032955983242
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
      value: 65.54472091459314
    - type: f1
      value: 61.47903120066317
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
      value: 61.45595158036314
    - type: f1
      value: 58.0891846024637
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
      value: 65.47074646940149
    - type: f1
      value: 62.84830858877575
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
      value: 58.046402151983855
    - type: f1
      value: 55.269074430533195
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
      value: 64.06523201075991
    - type: f1
      value: 61.35339643021369
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
      value: 60.954942837928726
    - type: f1
      value: 57.07035922704846
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
      value: 57.404169468728995
    - type: f1
      value: 53.94259011839138
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
      value: 64.16610625420309
    - type: f1
      value: 61.337103431499365
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
      value: 52.262945527908535
    - type: f1
      value: 49.7610691598921
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
      value: 65.54472091459314
    - type: f1
      value: 63.469099018440154
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
      value: 68.22797579018157
    - type: f1
      value: 64.89098471083001
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
      value: 50.847343644922674
    - type: f1
      value: 47.8536963168393
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
      value: 48.45326160053799
    - type: f1
      value: 46.370078045805556
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
      value: 42.83120376597175
    - type: f1
      value: 39.68948521599982
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
      value: 57.5084061869536
    - type: f1
      value: 53.961876160401545
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
      value: 63.7895090786819
    - type: f1
      value: 61.134223684676
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
      value: 54.98991257565569
    - type: f1
      value: 52.579862862826296
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
      value: 61.90316072629456
    - type: f1
      value: 58.203024538290336
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
      value: 57.09818426361802
    - type: f1
      value: 54.22718458445455
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
      value: 58.991257565568255
    - type: f1
      value: 55.84892781767421
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
      value: 55.901143241425686
    - type: f1
      value: 52.25264332199797
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
      value: 61.96368527236047
    - type: f1
      value: 58.927243876153454
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
      value: 65.64223268325489
    - type: f1
      value: 62.340453718379706
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
      value: 64.52589105581708
    - type: f1
      value: 61.661113187022174
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
      value: 66.84599865501009
    - type: f1
      value: 64.59342572873005
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
      value: 60.81035642232684
    - type: f1
      value: 57.5169089806797
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
      value: 65.75991930060525
    - type: f1
      value: 62.89531115787938
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
      value: 56.51647612642906
    - type: f1
      value: 54.33154780100043
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
      value: 57.985877605917956
    - type: f1
      value: 54.46187524463802
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
      value: 65.03026227303296
    - type: f1
      value: 62.34377392877748
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
      value: 53.567585743106925
    - type: f1
      value: 50.73770655983206
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
      value: 57.2595830531271
    - type: f1
      value: 53.657327291708626
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
      value: 57.82784129119032
    - type: f1
      value: 54.82518072665301
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
      value: 64.06859448554137
    - type: f1
      value: 63.00185280500495
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
      value: 58.91055817081371
    - type: f1
      value: 55.54116301224262
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
      value: 63.54404841963686
    - type: f1
      value: 59.57650946030184
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
      value: 59.27706792199059
    - type: f1
      value: 56.50010066083435
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
      value: 64.0719569603228
    - type: f1
      value: 61.817075925647956
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
      value: 68.23806321452591
    - type: f1
      value: 65.24917026029749
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
      value: 62.53530598520511
    - type: f1
      value: 61.71131132295768
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
      value: 63.04303967720243
    - type: f1
      value: 60.3950085685985
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
      value: 56.83591123066578
    - type: f1
      value: 54.95059828830849
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
      value: 59.62340282447881
    - type: f1
      value: 59.525159996498225
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
      value: 60.85406859448555
    - type: f1
      value: 59.129299095681276
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
      value: 62.76731674512441
    - type: f1
      value: 61.159560612627715
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
      value: 50.181573638197705
    - type: f1
      value: 46.98422176289957
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
      value: 68.92737054472092
    - type: f1
      value: 67.69135611952979
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
      value: 69.18964357767318
    - type: f1
      value: 68.46106138186214
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
      value: 67.0712844653665
    - type: f1
      value: 66.75545422473901
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
      value: 74.4754539340955
    - type: f1
      value: 74.38427146553252
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
      value: 69.82515131136518
    - type: f1
      value: 69.63516462173847
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
      value: 68.70880968392737
    - type: f1
      value: 67.45420662567926
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
      value: 65.95494283792871
    - type: f1
      value: 65.06191009049222
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
      value: 68.75924680564896
    - type: f1
      value: 68.30833379585945
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
      value: 63.806321452589096
    - type: f1
      value: 63.273048243765054
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
      value: 67.68997982515133
    - type: f1
      value: 66.54703855381324
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
      value: 66.46940147948891
    - type: f1
      value: 65.91017343463396
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
      value: 59.49899125756556
    - type: f1
      value: 57.90333469917769
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
      value: 67.9219905850706
    - type: f1
      value: 67.23169403762938
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
      value: 56.486213853396094
    - type: f1
      value: 54.85282355583758
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
      value: 69.04169468728985
    - type: f1
      value: 68.83833333320462
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
      value: 73.88702084734365
    - type: f1
      value: 74.04474735232299
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
      value: 56.63416274377943
    - type: f1
      value: 55.11332211687954
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
      value: 52.23604572965702
    - type: f1
      value: 50.86529813991055
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
      value: 46.62407531943511
    - type: f1
      value: 43.63485467164535
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
      value: 59.15601882985878
    - type: f1
      value: 57.522837510959924
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
      value: 69.84532616005382
    - type: f1
      value: 69.60021127179697
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
      value: 56.65770006724949
    - type: f1
      value: 55.84219135523227
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
      value: 66.53665097511768
    - type: f1
      value: 65.09087787792639
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
      value: 59.31405514458642
    - type: f1
      value: 58.06135303831491
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
      value: 64.88231338264964
    - type: f1
      value: 62.751099407787926
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
      value: 58.86012104909213
    - type: f1
      value: 56.29118323058282
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
      value: 67.37390719569602
    - type: f1
      value: 66.27922244885102
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
      value: 70.8675184936113
    - type: f1
      value: 70.22146529932019
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
      value: 68.2212508406187
    - type: f1
      value: 67.77454802056282
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
      value: 68.18090114324143
    - type: f1
      value: 68.03737625431621
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
      value: 64.65030262273034
    - type: f1
      value: 63.792945486912856
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
      value: 69.48217888365838
    - type: f1
      value: 69.96028997292197
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
      value: 60.17821116341627
    - type: f1
      value: 59.3935969827171
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
      value: 62.86146603900471
    - type: f1
      value: 60.133692735032376
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
      value: 70.89441829186282
    - type: f1
      value: 70.03064076194089
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
      value: 58.15063887020847
    - type: f1
      value: 56.23326278499678
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
      value: 59.43846671149966
    - type: f1
      value: 57.70440450281974
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
      value: 60.8507061197041
    - type: f1
      value: 59.22916396061171
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
      value: 70.65568258238063
    - type: f1
      value: 69.90736239440633
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
      value: 60.8843308675185
    - type: f1
      value: 59.30332663713599
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
      value: 68.05312710154674
    - type: f1
      value: 67.44024062594775
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
      value: 62.111634162743776
    - type: f1
      value: 60.89083013084519
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
      value: 67.44115669132482
    - type: f1
      value: 67.92227541674552
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
      value: 74.4687289845326
    - type: f1
      value: 74.16376793486025
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
      value: 68.31876260928043
    - type: f1
      value: 68.5246745215607
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
      value: 30.90431696479766
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
      value: 27.259158476693774
  - task:
      type: Reranking
    dataset:
      name: MTEB MindSmallReranking
      type: mteb/mind_small
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 30.28445330838555
    - type: mrr
      value: 31.15758529581164
  - task:
      type: Retrieval
    dataset:
      name: MTEB NFCorpus
      type: nfcorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.353
    - type: map_at_10
      value: 11.565
    - type: map_at_100
      value: 14.097000000000001
    - type: map_at_1000
      value: 15.354999999999999
    - type: map_at_3
      value: 8.749
    - type: map_at_5
      value: 9.974
    - type: mrr_at_1
      value: 42.105
    - type: mrr_at_10
      value: 50.589
    - type: mrr_at_100
      value: 51.187000000000005
    - type: mrr_at_1000
      value: 51.233
    - type: mrr_at_3
      value: 48.246
    - type: mrr_at_5
      value: 49.546
    - type: ndcg_at_1
      value: 40.402
    - type: ndcg_at_10
      value: 31.009999999999998
    - type: ndcg_at_100
      value: 28.026
    - type: ndcg_at_1000
      value: 36.905
    - type: ndcg_at_3
      value: 35.983
    - type: ndcg_at_5
      value: 33.764
    - type: precision_at_1
      value: 42.105
    - type: precision_at_10
      value: 22.786
    - type: precision_at_100
      value: 6.916
    - type: precision_at_1000
      value: 1.981
    - type: precision_at_3
      value: 33.333
    - type: precision_at_5
      value: 28.731
    - type: recall_at_1
      value: 5.353
    - type: recall_at_10
      value: 15.039
    - type: recall_at_100
      value: 27.348
    - type: recall_at_1000
      value: 59.453
    - type: recall_at_3
      value: 9.792
    - type: recall_at_5
      value: 11.882
  - task:
      type: Retrieval
    dataset:
      name: MTEB NQ
      type: nq
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 33.852
    - type: map_at_10
      value: 48.924
    - type: map_at_100
      value: 49.854
    - type: map_at_1000
      value: 49.886
    - type: map_at_3
      value: 44.9
    - type: map_at_5
      value: 47.387
    - type: mrr_at_1
      value: 38.035999999999994
    - type: mrr_at_10
      value: 51.644
    - type: mrr_at_100
      value: 52.339
    - type: mrr_at_1000
      value: 52.35999999999999
    - type: mrr_at_3
      value: 48.421
    - type: mrr_at_5
      value: 50.468999999999994
    - type: ndcg_at_1
      value: 38.007000000000005
    - type: ndcg_at_10
      value: 56.293000000000006
    - type: ndcg_at_100
      value: 60.167
    - type: ndcg_at_1000
      value: 60.916000000000004
    - type: ndcg_at_3
      value: 48.903999999999996
    - type: ndcg_at_5
      value: 52.978
    - type: precision_at_1
      value: 38.007000000000005
    - type: precision_at_10
      value: 9.041
    - type: precision_at_100
      value: 1.1199999999999999
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 22.084
    - type: precision_at_5
      value: 15.608
    - type: recall_at_1
      value: 33.852
    - type: recall_at_10
      value: 75.893
    - type: recall_at_100
      value: 92.589
    - type: recall_at_1000
      value: 98.153
    - type: recall_at_3
      value: 56.969
    - type: recall_at_5
      value: 66.283
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
      value: 69.174
    - type: map_at_10
      value: 82.891
    - type: map_at_100
      value: 83.545
    - type: map_at_1000
      value: 83.56700000000001
    - type: map_at_3
      value: 79.944
    - type: map_at_5
      value: 81.812
    - type: mrr_at_1
      value: 79.67999999999999
    - type: mrr_at_10
      value: 86.279
    - type: mrr_at_100
      value: 86.39
    - type: mrr_at_1000
      value: 86.392
    - type: mrr_at_3
      value: 85.21
    - type: mrr_at_5
      value: 85.92999999999999
    - type: ndcg_at_1
      value: 79.69000000000001
    - type: ndcg_at_10
      value: 86.929
    - type: ndcg_at_100
      value: 88.266
    - type: ndcg_at_1000
      value: 88.428
    - type: ndcg_at_3
      value: 83.899
    - type: ndcg_at_5
      value: 85.56700000000001
    - type: precision_at_1
      value: 79.69000000000001
    - type: precision_at_10
      value: 13.161000000000001
    - type: precision_at_100
      value: 1.513
    - type: precision_at_1000
      value: 0.156
    - type: precision_at_3
      value: 36.603
    - type: precision_at_5
      value: 24.138
    - type: recall_at_1
      value: 69.174
    - type: recall_at_10
      value: 94.529
    - type: recall_at_100
      value: 99.15
    - type: recall_at_1000
      value: 99.925
    - type: recall_at_3
      value: 85.86200000000001
    - type: recall_at_5
      value: 90.501
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
      value: 39.13064340585255
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
      value: 58.97884249325877
  - task:
      type: Retrieval
    dataset:
      name: MTEB SCIDOCS
      type: scidocs
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 3.4680000000000004
    - type: map_at_10
      value: 7.865
    - type: map_at_100
      value: 9.332
    - type: map_at_1000
      value: 9.587
    - type: map_at_3
      value: 5.800000000000001
    - type: map_at_5
      value: 6.8790000000000004
    - type: mrr_at_1
      value: 17.0
    - type: mrr_at_10
      value: 25.629
    - type: mrr_at_100
      value: 26.806
    - type: mrr_at_1000
      value: 26.889000000000003
    - type: mrr_at_3
      value: 22.8
    - type: mrr_at_5
      value: 24.26
    - type: ndcg_at_1
      value: 17.0
    - type: ndcg_at_10
      value: 13.895
    - type: ndcg_at_100
      value: 20.491999999999997
    - type: ndcg_at_1000
      value: 25.759999999999998
    - type: ndcg_at_3
      value: 13.347999999999999
    - type: ndcg_at_5
      value: 11.61
    - type: precision_at_1
      value: 17.0
    - type: precision_at_10
      value: 7.090000000000001
    - type: precision_at_100
      value: 1.669
    - type: precision_at_1000
      value: 0.294
    - type: precision_at_3
      value: 12.3
    - type: precision_at_5
      value: 10.02
    - type: recall_at_1
      value: 3.4680000000000004
    - type: recall_at_10
      value: 14.363000000000001
    - type: recall_at_100
      value: 33.875
    - type: recall_at_1000
      value: 59.711999999999996
    - type: recall_at_3
      value: 7.483
    - type: recall_at_5
      value: 10.173
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
      value: 83.04084311714061
    - type: cos_sim_spearman
      value: 77.51342467443078
    - type: euclidean_pearson
      value: 80.0321166028479
    - type: euclidean_spearman
      value: 77.29249114733226
    - type: manhattan_pearson
      value: 80.03105964262431
    - type: manhattan_spearman
      value: 77.22373689514794
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
      value: 84.1680158034387
    - type: cos_sim_spearman
      value: 76.55983344071117
    - type: euclidean_pearson
      value: 79.75266678300143
    - type: euclidean_spearman
      value: 75.34516823467025
    - type: manhattan_pearson
      value: 79.75959151517357
    - type: manhattan_spearman
      value: 75.42330344141912
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
      value: 76.48898993209346
    - type: cos_sim_spearman
      value: 76.96954120323366
    - type: euclidean_pearson
      value: 76.94139109279668
    - type: euclidean_spearman
      value: 76.85860283201711
    - type: manhattan_pearson
      value: 76.6944095091912
    - type: manhattan_spearman
      value: 76.61096912972553
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
      value: 77.85082366246944
    - type: cos_sim_spearman
      value: 75.52053350101731
    - type: euclidean_pearson
      value: 77.1165845070926
    - type: euclidean_spearman
      value: 75.31216065884388
    - type: manhattan_pearson
      value: 77.06193941833494
    - type: manhattan_spearman
      value: 75.31003701700112
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
      value: 86.36305246526497
    - type: cos_sim_spearman
      value: 87.11704613927415
    - type: euclidean_pearson
      value: 86.04199125810939
    - type: euclidean_spearman
      value: 86.51117572414263
    - type: manhattan_pearson
      value: 86.0805106816633
    - type: manhattan_spearman
      value: 86.52798366512229
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
      value: 82.18536255599724
    - type: cos_sim_spearman
      value: 83.63377151025418
    - type: euclidean_pearson
      value: 83.24657467993141
    - type: euclidean_spearman
      value: 84.02751481993825
    - type: manhattan_pearson
      value: 83.11941806582371
    - type: manhattan_spearman
      value: 83.84251281019304
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
      value: 78.95816528475514
    - type: cos_sim_spearman
      value: 78.86607380120462
    - type: euclidean_pearson
      value: 78.51268699230545
    - type: euclidean_spearman
      value: 79.11649316502229
    - type: manhattan_pearson
      value: 78.32367302808157
    - type: manhattan_spearman
      value: 78.90277699624637
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
      value: 72.89126914997624
    - type: cos_sim_spearman
      value: 73.0296921832678
    - type: euclidean_pearson
      value: 71.50385903677738
    - type: euclidean_spearman
      value: 73.13368899716289
    - type: manhattan_pearson
      value: 71.47421463379519
    - type: manhattan_spearman
      value: 73.03383242946575
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
      value: 59.22923684492637
    - type: cos_sim_spearman
      value: 57.41013211368396
    - type: euclidean_pearson
      value: 61.21107388080905
    - type: euclidean_spearman
      value: 60.07620768697254
    - type: manhattan_pearson
      value: 59.60157142786555
    - type: manhattan_spearman
      value: 59.14069604103739
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
      value: 76.24345978774299
    - type: cos_sim_spearman
      value: 77.24225743830719
    - type: euclidean_pearson
      value: 76.66226095469165
    - type: euclidean_spearman
      value: 77.60708820493146
    - type: manhattan_pearson
      value: 76.05303324760429
    - type: manhattan_spearman
      value: 76.96353149912348
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
      value: 85.50879160160852
    - type: cos_sim_spearman
      value: 86.43594662965224
    - type: euclidean_pearson
      value: 86.06846012826577
    - type: euclidean_spearman
      value: 86.02041395794136
    - type: manhattan_pearson
      value: 86.10916255616904
    - type: manhattan_spearman
      value: 86.07346068198953
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
      value: 58.39803698977196
    - type: cos_sim_spearman
      value: 55.96910950423142
    - type: euclidean_pearson
      value: 58.17941175613059
    - type: euclidean_spearman
      value: 55.03019330522745
    - type: manhattan_pearson
      value: 57.333358138183286
    - type: manhattan_spearman
      value: 54.04614023149965
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
      value: 70.98304089637197
    - type: cos_sim_spearman
      value: 72.44071656215888
    - type: euclidean_pearson
      value: 72.19224359033983
    - type: euclidean_spearman
      value: 73.89871188913025
    - type: manhattan_pearson
      value: 71.21098311547406
    - type: manhattan_spearman
      value: 72.93405764824821
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
      value: 85.99792397466308
    - type: cos_sim_spearman
      value: 84.83824377879495
    - type: euclidean_pearson
      value: 85.70043288694438
    - type: euclidean_spearman
      value: 84.70627558703686
    - type: manhattan_pearson
      value: 85.89570850150801
    - type: manhattan_spearman
      value: 84.95806105313007
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
      value: 72.21850322994712
    - type: cos_sim_spearman
      value: 72.28669398117248
    - type: euclidean_pearson
      value: 73.40082510412948
    - type: euclidean_spearman
      value: 73.0326539281865
    - type: manhattan_pearson
      value: 71.8659633964841
    - type: manhattan_spearman
      value: 71.57817425823303
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
      value: 75.80921368595645
    - type: cos_sim_spearman
      value: 77.33209091229315
    - type: euclidean_pearson
      value: 76.53159540154829
    - type: euclidean_spearman
      value: 78.17960842810093
    - type: manhattan_pearson
      value: 76.13530186637601
    - type: manhattan_spearman
      value: 78.00701437666875
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
      value: 74.74980608267349
    - type: cos_sim_spearman
      value: 75.37597374318821
    - type: euclidean_pearson
      value: 74.90506081911661
    - type: euclidean_spearman
      value: 75.30151613124521
    - type: manhattan_pearson
      value: 74.62642745918002
    - type: manhattan_spearman
      value: 75.18619716592303
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
      value: 59.632662289205584
    - type: cos_sim_spearman
      value: 60.938543391610914
    - type: euclidean_pearson
      value: 62.113200529767056
    - type: euclidean_spearman
      value: 61.410312633261164
    - type: manhattan_pearson
      value: 61.75494698945686
    - type: manhattan_spearman
      value: 60.92726195322362
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
      value: 45.283470551557244
    - type: cos_sim_spearman
      value: 53.44833015864201
    - type: euclidean_pearson
      value: 41.17892011120893
    - type: euclidean_spearman
      value: 53.81441383126767
    - type: manhattan_pearson
      value: 41.17482200420659
    - type: manhattan_spearman
      value: 53.82180269276363
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
      value: 60.5069165306236
    - type: cos_sim_spearman
      value: 66.87803259033826
    - type: euclidean_pearson
      value: 63.5428979418236
    - type: euclidean_spearman
      value: 66.9293576586897
    - type: manhattan_pearson
      value: 63.59789526178922
    - type: manhattan_spearman
      value: 66.86555009875066
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
      value: 28.23026196280264
    - type: cos_sim_spearman
      value: 35.79397812652861
    - type: euclidean_pearson
      value: 17.828102102767353
    - type: euclidean_spearman
      value: 35.721501145568894
    - type: manhattan_pearson
      value: 17.77134274219677
    - type: manhattan_spearman
      value: 35.98107902846267
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
      value: 56.51946541393812
    - type: cos_sim_spearman
      value: 63.714686006214485
    - type: euclidean_pearson
      value: 58.32104651305898
    - type: euclidean_spearman
      value: 62.237110895702216
    - type: manhattan_pearson
      value: 58.579416468759185
    - type: manhattan_spearman
      value: 62.459738981727
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
      value: 48.76009839569795
    - type: cos_sim_spearman
      value: 56.65188431953149
    - type: euclidean_pearson
      value: 50.997682160915595
    - type: euclidean_spearman
      value: 55.99910008818135
    - type: manhattan_pearson
      value: 50.76220659606342
    - type: manhattan_spearman
      value: 55.517347595391456
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
      value: 51.232731157702425
    - type: cos_sim_spearman
      value: 59.89531877658345
    - type: euclidean_pearson
      value: 49.937914570348376
    - type: euclidean_spearman
      value: 60.220905659334036
    - type: manhattan_pearson
      value: 50.00987996844193
    - type: manhattan_spearman
      value: 60.081341480977926
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
      value: 54.717524559088005
    - type: cos_sim_spearman
      value: 66.83570886252286
    - type: euclidean_pearson
      value: 58.41338625505467
    - type: euclidean_spearman
      value: 66.68991427704938
    - type: manhattan_pearson
      value: 58.78638572916807
    - type: manhattan_spearman
      value: 66.58684161046335
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
      value: 73.2962042954962
    - type: cos_sim_spearman
      value: 76.58255504852025
    - type: euclidean_pearson
      value: 75.70983192778257
    - type: euclidean_spearman
      value: 77.4547684870542
    - type: manhattan_pearson
      value: 75.75565853870485
    - type: manhattan_spearman
      value: 76.90208974949428
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
      value: 54.47396266924846
    - type: cos_sim_spearman
      value: 56.492267162048606
    - type: euclidean_pearson
      value: 55.998505203070195
    - type: euclidean_spearman
      value: 56.46447012960222
    - type: manhattan_pearson
      value: 54.873172394430995
    - type: manhattan_spearman
      value: 56.58111534551218
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
      value: 69.87177267688686
    - type: cos_sim_spearman
      value: 74.57160943395763
    - type: euclidean_pearson
      value: 70.88330406826788
    - type: euclidean_spearman
      value: 74.29767636038422
    - type: manhattan_pearson
      value: 71.38245248369536
    - type: manhattan_spearman
      value: 74.53102232732175
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
      value: 72.80225656959544
    - type: cos_sim_spearman
      value: 76.52646173725735
    - type: euclidean_pearson
      value: 73.95710720200799
    - type: euclidean_spearman
      value: 76.54040031984111
    - type: manhattan_pearson
      value: 73.89679971946774
    - type: manhattan_spearman
      value: 76.60886958161574
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
      value: 70.70844249898789
    - type: cos_sim_spearman
      value: 72.68571783670241
    - type: euclidean_pearson
      value: 72.38800772441031
    - type: euclidean_spearman
      value: 72.86804422703312
    - type: manhattan_pearson
      value: 71.29840508203515
    - type: manhattan_spearman
      value: 71.86264441749513
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
      value: 58.647478923935694
    - type: cos_sim_spearman
      value: 63.74453623540931
    - type: euclidean_pearson
      value: 59.60138032437505
    - type: euclidean_spearman
      value: 63.947930832166065
    - type: manhattan_pearson
      value: 58.59735509491861
    - type: manhattan_spearman
      value: 62.082503844627404
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
      value: 65.8722516867162
    - type: cos_sim_spearman
      value: 71.81208592523012
    - type: euclidean_pearson
      value: 67.95315252165956
    - type: euclidean_spearman
      value: 73.00749822046009
    - type: manhattan_pearson
      value: 68.07884688638924
    - type: manhattan_spearman
      value: 72.34210325803069
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
      value: 54.5405814240949
    - type: cos_sim_spearman
      value: 60.56838649023775
    - type: euclidean_pearson
      value: 53.011731611314104
    - type: euclidean_spearman
      value: 58.533194841668426
    - type: manhattan_pearson
      value: 53.623067729338494
    - type: manhattan_spearman
      value: 58.018756154446926
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
      value: 13.611046866216112
    - type: cos_sim_spearman
      value: 28.238192909158492
    - type: euclidean_pearson
      value: 22.16189199885129
    - type: euclidean_spearman
      value: 35.012895679076564
    - type: manhattan_pearson
      value: 21.969771178698387
    - type: manhattan_spearman
      value: 32.456985088607475
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
      value: 74.58077407011655
    - type: cos_sim_spearman
      value: 84.51542547285167
    - type: euclidean_pearson
      value: 74.64613843596234
    - type: euclidean_spearman
      value: 84.51542547285167
    - type: manhattan_pearson
      value: 75.15335973101396
    - type: manhattan_spearman
      value: 84.51542547285167
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
      value: 82.0739825531578
    - type: cos_sim_spearman
      value: 84.01057479311115
    - type: euclidean_pearson
      value: 83.85453227433344
    - type: euclidean_spearman
      value: 84.01630226898655
    - type: manhattan_pearson
      value: 83.75323603028978
    - type: manhattan_spearman
      value: 83.89677983727685
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
      value: 78.12945623123957
    - type: mrr
      value: 93.87738713719106
  - task:
      type: Retrieval
    dataset:
      name: MTEB SciFact
      type: scifact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 52.983000000000004
    - type: map_at_10
      value: 62.946000000000005
    - type: map_at_100
      value: 63.514
    - type: map_at_1000
      value: 63.554
    - type: map_at_3
      value: 60.183
    - type: map_at_5
      value: 61.672000000000004
    - type: mrr_at_1
      value: 55.667
    - type: mrr_at_10
      value: 64.522
    - type: mrr_at_100
      value: 64.957
    - type: mrr_at_1000
      value: 64.995
    - type: mrr_at_3
      value: 62.388999999999996
    - type: mrr_at_5
      value: 63.639
    - type: ndcg_at_1
      value: 55.667
    - type: ndcg_at_10
      value: 67.704
    - type: ndcg_at_100
      value: 70.299
    - type: ndcg_at_1000
      value: 71.241
    - type: ndcg_at_3
      value: 62.866
    - type: ndcg_at_5
      value: 65.16999999999999
    - type: precision_at_1
      value: 55.667
    - type: precision_at_10
      value: 9.033
    - type: precision_at_100
      value: 1.053
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 24.444
    - type: precision_at_5
      value: 16.133
    - type: recall_at_1
      value: 52.983000000000004
    - type: recall_at_10
      value: 80.656
    - type: recall_at_100
      value: 92.5
    - type: recall_at_1000
      value: 99.667
    - type: recall_at_3
      value: 67.744
    - type: recall_at_5
      value: 73.433
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
      value: 99.72772277227723
    - type: cos_sim_ap
      value: 92.17845897992215
    - type: cos_sim_f1
      value: 85.9746835443038
    - type: cos_sim_precision
      value: 87.07692307692308
    - type: cos_sim_recall
      value: 84.89999999999999
    - type: dot_accuracy
      value: 99.3039603960396
    - type: dot_ap
      value: 60.70244020124878
    - type: dot_f1
      value: 59.92742353551063
    - type: dot_precision
      value: 62.21743810548978
    - type: dot_recall
      value: 57.8
    - type: euclidean_accuracy
      value: 99.71683168316832
    - type: euclidean_ap
      value: 91.53997039964659
    - type: euclidean_f1
      value: 84.88372093023257
    - type: euclidean_precision
      value: 90.02242152466367
    - type: euclidean_recall
      value: 80.30000000000001
    - type: manhattan_accuracy
      value: 99.72376237623763
    - type: manhattan_ap
      value: 91.80756777790289
    - type: manhattan_f1
      value: 85.48468106479157
    - type: manhattan_precision
      value: 85.8728557013118
    - type: manhattan_recall
      value: 85.1
    - type: max_accuracy
      value: 99.72772277227723
    - type: max_ap
      value: 92.17845897992215
    - type: max_f1
      value: 85.9746835443038
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
      value: 53.52464042600003
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
      value: 32.071631948736
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
      value: 49.19552407604654
    - type: mrr
      value: 49.95269130379425
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
      value: 29.345293033095427
    - type: cos_sim_spearman
      value: 29.976931423258403
    - type: dot_pearson
      value: 27.047078008958408
    - type: dot_spearman
      value: 27.75894368380218
  - task:
      type: Retrieval
    dataset:
      name: MTEB TRECCOVID
      type: trec-covid
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.22
    - type: map_at_10
      value: 1.706
    - type: map_at_100
      value: 9.634
    - type: map_at_1000
      value: 23.665
    - type: map_at_3
      value: 0.5950000000000001
    - type: map_at_5
      value: 0.95
    - type: mrr_at_1
      value: 86.0
    - type: mrr_at_10
      value: 91.8
    - type: mrr_at_100
      value: 91.8
    - type: mrr_at_1000
      value: 91.8
    - type: mrr_at_3
      value: 91.0
    - type: mrr_at_5
      value: 91.8
    - type: ndcg_at_1
      value: 80.0
    - type: ndcg_at_10
      value: 72.573
    - type: ndcg_at_100
      value: 53.954
    - type: ndcg_at_1000
      value: 47.760999999999996
    - type: ndcg_at_3
      value: 76.173
    - type: ndcg_at_5
      value: 75.264
    - type: precision_at_1
      value: 86.0
    - type: precision_at_10
      value: 76.4
    - type: precision_at_100
      value: 55.50000000000001
    - type: precision_at_1000
      value: 21.802
    - type: precision_at_3
      value: 81.333
    - type: precision_at_5
      value: 80.4
    - type: recall_at_1
      value: 0.22
    - type: recall_at_10
      value: 1.925
    - type: recall_at_100
      value: 12.762
    - type: recall_at_1000
      value: 44.946000000000005
    - type: recall_at_3
      value: 0.634
    - type: recall_at_5
      value: 1.051
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (sqi-eng)
      type: mteb/tatoeba-bitext-mining
      config: sqi-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.0
    - type: f1
      value: 88.55666666666666
    - type: precision
      value: 87.46166666666667
    - type: recall
      value: 91.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fry-eng)
      type: mteb/tatoeba-bitext-mining
      config: fry-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 57.22543352601156
    - type: f1
      value: 51.03220478943021
    - type: precision
      value: 48.8150289017341
    - type: recall
      value: 57.22543352601156
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kur-eng)
      type: mteb/tatoeba-bitext-mining
      config: kur-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 46.58536585365854
    - type: f1
      value: 39.66870798578116
    - type: precision
      value: 37.416085946573745
    - type: recall
      value: 46.58536585365854
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tur-eng)
      type: mteb/tatoeba-bitext-mining
      config: tur-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.7
    - type: f1
      value: 86.77999999999999
    - type: precision
      value: 85.45333333333332
    - type: recall
      value: 89.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (deu-eng)
      type: mteb/tatoeba-bitext-mining
      config: deu-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.39999999999999
    - type: f1
      value: 96.58333333333331
    - type: precision
      value: 96.2
    - type: recall
      value: 97.39999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nld-eng)
      type: mteb/tatoeba-bitext-mining
      config: nld-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.4
    - type: f1
      value: 90.3
    - type: precision
      value: 89.31666666666668
    - type: recall
      value: 92.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ron-eng)
      type: mteb/tatoeba-bitext-mining
      config: ron-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.9
    - type: f1
      value: 83.67190476190476
    - type: precision
      value: 82.23333333333332
    - type: recall
      value: 86.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ang-eng)
      type: mteb/tatoeba-bitext-mining
      config: ang-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 50.0
    - type: f1
      value: 42.23229092632078
    - type: precision
      value: 39.851634683724235
    - type: recall
      value: 50.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ido-eng)
      type: mteb/tatoeba-bitext-mining
      config: ido-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 76.3
    - type: f1
      value: 70.86190476190477
    - type: precision
      value: 68.68777777777777
    - type: recall
      value: 76.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (jav-eng)
      type: mteb/tatoeba-bitext-mining
      config: jav-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 57.073170731707314
    - type: f1
      value: 50.658958927251604
    - type: precision
      value: 48.26480836236933
    - type: recall
      value: 57.073170731707314
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (isl-eng)
      type: mteb/tatoeba-bitext-mining
      config: isl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 68.2
    - type: f1
      value: 62.156507936507936
    - type: precision
      value: 59.84964285714286
    - type: recall
      value: 68.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (slv-eng)
      type: mteb/tatoeba-bitext-mining
      config: slv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.52126366950182
    - type: f1
      value: 72.8496210148701
    - type: precision
      value: 70.92171498003819
    - type: recall
      value: 77.52126366950182
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cym-eng)
      type: mteb/tatoeba-bitext-mining
      config: cym-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 70.78260869565217
    - type: f1
      value: 65.32422360248447
    - type: precision
      value: 63.063067367415194
    - type: recall
      value: 70.78260869565217
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kaz-eng)
      type: mteb/tatoeba-bitext-mining
      config: kaz-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 78.43478260869566
    - type: f1
      value: 73.02608695652172
    - type: precision
      value: 70.63768115942028
    - type: recall
      value: 78.43478260869566
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (est-eng)
      type: mteb/tatoeba-bitext-mining
      config: est-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 60.9
    - type: f1
      value: 55.309753694581275
    - type: precision
      value: 53.130476190476195
    - type: recall
      value: 60.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (heb-eng)
      type: mteb/tatoeba-bitext-mining
      config: heb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 72.89999999999999
    - type: f1
      value: 67.92023809523809
    - type: precision
      value: 65.82595238095237
    - type: recall
      value: 72.89999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (gla-eng)
      type: mteb/tatoeba-bitext-mining
      config: gla-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 46.80337756332931
    - type: f1
      value: 39.42174900558496
    - type: precision
      value: 36.97101116280851
    - type: recall
      value: 46.80337756332931
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mar-eng)
      type: mteb/tatoeba-bitext-mining
      config: mar-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.8
    - type: f1
      value: 86.79
    - type: precision
      value: 85.375
    - type: recall
      value: 89.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lat-eng)
      type: mteb/tatoeba-bitext-mining
      config: lat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 47.199999999999996
    - type: f1
      value: 39.95484348984349
    - type: precision
      value: 37.561071428571424
    - type: recall
      value: 47.199999999999996
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bel-eng)
      type: mteb/tatoeba-bitext-mining
      config: bel-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.8
    - type: f1
      value: 84.68190476190475
    - type: precision
      value: 83.275
    - type: recall
      value: 87.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pms-eng)
      type: mteb/tatoeba-bitext-mining
      config: pms-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 48.76190476190476
    - type: f1
      value: 42.14965986394558
    - type: precision
      value: 39.96743626743626
    - type: recall
      value: 48.76190476190476
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (gle-eng)
      type: mteb/tatoeba-bitext-mining
      config: gle-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 66.10000000000001
    - type: f1
      value: 59.58580086580086
    - type: precision
      value: 57.150238095238095
    - type: recall
      value: 66.10000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pes-eng)
      type: mteb/tatoeba-bitext-mining
      config: pes-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.3
    - type: f1
      value: 84.0
    - type: precision
      value: 82.48666666666666
    - type: recall
      value: 87.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nob-eng)
      type: mteb/tatoeba-bitext-mining
      config: nob-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.4
    - type: f1
      value: 87.79523809523809
    - type: precision
      value: 86.6
    - type: recall
      value: 90.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bul-eng)
      type: mteb/tatoeba-bitext-mining
      config: bul-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.0
    - type: f1
      value: 83.81
    - type: precision
      value: 82.36666666666666
    - type: recall
      value: 87.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cbk-eng)
      type: mteb/tatoeba-bitext-mining
      config: cbk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 63.9
    - type: f1
      value: 57.76533189033189
    - type: precision
      value: 55.50595238095239
    - type: recall
      value: 63.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hun-eng)
      type: mteb/tatoeba-bitext-mining
      config: hun-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 76.1
    - type: f1
      value: 71.83690476190478
    - type: precision
      value: 70.04928571428573
    - type: recall
      value: 76.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (uig-eng)
      type: mteb/tatoeba-bitext-mining
      config: uig-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 66.3
    - type: f1
      value: 59.32626984126984
    - type: precision
      value: 56.62535714285713
    - type: recall
      value: 66.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (rus-eng)
      type: mteb/tatoeba-bitext-mining
      config: rus-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.60000000000001
    - type: f1
      value: 87.96333333333334
    - type: precision
      value: 86.73333333333333
    - type: recall
      value: 90.60000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (spa-eng)
      type: mteb/tatoeba-bitext-mining
      config: spa-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.10000000000001
    - type: f1
      value: 91.10000000000001
    - type: precision
      value: 90.16666666666666
    - type: recall
      value: 93.10000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hye-eng)
      type: mteb/tatoeba-bitext-mining
      config: hye-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.71428571428571
    - type: f1
      value: 82.29142600436403
    - type: precision
      value: 80.8076626877166
    - type: recall
      value: 85.71428571428571
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tel-eng)
      type: mteb/tatoeba-bitext-mining
      config: tel-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.88888888888889
    - type: f1
      value: 85.7834757834758
    - type: precision
      value: 84.43732193732193
    - type: recall
      value: 88.88888888888889
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (afr-eng)
      type: mteb/tatoeba-bitext-mining
      config: afr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.5
    - type: f1
      value: 85.67190476190476
    - type: precision
      value: 84.43333333333332
    - type: recall
      value: 88.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mon-eng)
      type: mteb/tatoeba-bitext-mining
      config: mon-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 82.72727272727273
    - type: f1
      value: 78.21969696969695
    - type: precision
      value: 76.18181818181819
    - type: recall
      value: 82.72727272727273
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (arz-eng)
      type: mteb/tatoeba-bitext-mining
      config: arz-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 61.0062893081761
    - type: f1
      value: 55.13976240391334
    - type: precision
      value: 52.92112499659669
    - type: recall
      value: 61.0062893081761
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hrv-eng)
      type: mteb/tatoeba-bitext-mining
      config: hrv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.5
    - type: f1
      value: 86.86666666666666
    - type: precision
      value: 85.69166666666668
    - type: recall
      value: 89.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nov-eng)
      type: mteb/tatoeba-bitext-mining
      config: nov-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 73.54085603112841
    - type: f1
      value: 68.56031128404669
    - type: precision
      value: 66.53047989623866
    - type: recall
      value: 73.54085603112841
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (gsw-eng)
      type: mteb/tatoeba-bitext-mining
      config: gsw-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 43.58974358974359
    - type: f1
      value: 36.45299145299145
    - type: precision
      value: 33.81155881155882
    - type: recall
      value: 43.58974358974359
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nds-eng)
      type: mteb/tatoeba-bitext-mining
      config: nds-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 59.599999999999994
    - type: f1
      value: 53.264689754689755
    - type: precision
      value: 50.869166666666665
    - type: recall
      value: 59.599999999999994
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ukr-eng)
      type: mteb/tatoeba-bitext-mining
      config: ukr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.2
    - type: f1
      value: 81.61666666666665
    - type: precision
      value: 80.02833333333335
    - type: recall
      value: 85.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (uzb-eng)
      type: mteb/tatoeba-bitext-mining
      config: uzb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 63.78504672897196
    - type: f1
      value: 58.00029669188548
    - type: precision
      value: 55.815809968847354
    - type: recall
      value: 63.78504672897196
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lit-eng)
      type: mteb/tatoeba-bitext-mining
      config: lit-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 66.5
    - type: f1
      value: 61.518333333333345
    - type: precision
      value: 59.622363699102834
    - type: recall
      value: 66.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ina-eng)
      type: mteb/tatoeba-bitext-mining
      config: ina-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.6
    - type: f1
      value: 85.60222222222221
    - type: precision
      value: 84.27916666666665
    - type: recall
      value: 88.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lfn-eng)
      type: mteb/tatoeba-bitext-mining
      config: lfn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 58.699999999999996
    - type: f1
      value: 52.732375957375965
    - type: precision
      value: 50.63214035964035
    - type: recall
      value: 58.699999999999996
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (zsm-eng)
      type: mteb/tatoeba-bitext-mining
      config: zsm-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.10000000000001
    - type: f1
      value: 89.99666666666667
    - type: precision
      value: 89.03333333333333
    - type: recall
      value: 92.10000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ita-eng)
      type: mteb/tatoeba-bitext-mining
      config: ita-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.10000000000001
    - type: f1
      value: 87.55666666666667
    - type: precision
      value: 86.36166666666668
    - type: recall
      value: 90.10000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cmn-eng)
      type: mteb/tatoeba-bitext-mining
      config: cmn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.4
    - type: f1
      value: 88.89000000000001
    - type: precision
      value: 87.71166666666666
    - type: recall
      value: 91.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lvs-eng)
      type: mteb/tatoeba-bitext-mining
      config: lvs-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 65.7
    - type: f1
      value: 60.67427750410509
    - type: precision
      value: 58.71785714285714
    - type: recall
      value: 65.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (glg-eng)
      type: mteb/tatoeba-bitext-mining
      config: glg-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.39999999999999
    - type: f1
      value: 81.93190476190475
    - type: precision
      value: 80.37833333333333
    - type: recall
      value: 85.39999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ceb-eng)
      type: mteb/tatoeba-bitext-mining
      config: ceb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 47.833333333333336
    - type: f1
      value: 42.006625781625786
    - type: precision
      value: 40.077380952380956
    - type: recall
      value: 47.833333333333336
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bre-eng)
      type: mteb/tatoeba-bitext-mining
      config: bre-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 10.4
    - type: f1
      value: 8.24465007215007
    - type: precision
      value: 7.664597069597071
    - type: recall
      value: 10.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ben-eng)
      type: mteb/tatoeba-bitext-mining
      config: ben-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 82.6
    - type: f1
      value: 77.76333333333334
    - type: precision
      value: 75.57833333333332
    - type: recall
      value: 82.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (swg-eng)
      type: mteb/tatoeba-bitext-mining
      config: swg-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 52.67857142857143
    - type: f1
      value: 44.302721088435376
    - type: precision
      value: 41.49801587301587
    - type: recall
      value: 52.67857142857143
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (arq-eng)
      type: mteb/tatoeba-bitext-mining
      config: arq-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 28.3205268935236
    - type: f1
      value: 22.426666605171157
    - type: precision
      value: 20.685900116470915
    - type: recall
      value: 28.3205268935236
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kab-eng)
      type: mteb/tatoeba-bitext-mining
      config: kab-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 22.7
    - type: f1
      value: 17.833970473970474
    - type: precision
      value: 16.407335164835164
    - type: recall
      value: 22.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fra-eng)
      type: mteb/tatoeba-bitext-mining
      config: fra-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.2
    - type: f1
      value: 89.92999999999999
    - type: precision
      value: 88.87
    - type: recall
      value: 92.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (por-eng)
      type: mteb/tatoeba-bitext-mining
      config: por-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.4
    - type: f1
      value: 89.25
    - type: precision
      value: 88.21666666666667
    - type: recall
      value: 91.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tat-eng)
      type: mteb/tatoeba-bitext-mining
      config: tat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 69.19999999999999
    - type: f1
      value: 63.38269841269841
    - type: precision
      value: 61.14773809523809
    - type: recall
      value: 69.19999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (oci-eng)
      type: mteb/tatoeba-bitext-mining
      config: oci-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 48.8
    - type: f1
      value: 42.839915639915645
    - type: precision
      value: 40.770287114845935
    - type: recall
      value: 48.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pol-eng)
      type: mteb/tatoeba-bitext-mining
      config: pol-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.8
    - type: f1
      value: 85.90666666666668
    - type: precision
      value: 84.54166666666666
    - type: recall
      value: 88.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (war-eng)
      type: mteb/tatoeba-bitext-mining
      config: war-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 46.6
    - type: f1
      value: 40.85892920804686
    - type: precision
      value: 38.838223114604695
    - type: recall
      value: 46.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (aze-eng)
      type: mteb/tatoeba-bitext-mining
      config: aze-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.0
    - type: f1
      value: 80.14190476190475
    - type: precision
      value: 78.45333333333333
    - type: recall
      value: 84.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (vie-eng)
      type: mteb/tatoeba-bitext-mining
      config: vie-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.5
    - type: f1
      value: 87.78333333333333
    - type: precision
      value: 86.5
    - type: recall
      value: 90.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nno-eng)
      type: mteb/tatoeba-bitext-mining
      config: nno-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 74.5
    - type: f1
      value: 69.48397546897547
    - type: precision
      value: 67.51869047619049
    - type: recall
      value: 74.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cha-eng)
      type: mteb/tatoeba-bitext-mining
      config: cha-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 32.846715328467155
    - type: f1
      value: 27.828177499710343
    - type: precision
      value: 26.63451511991658
    - type: recall
      value: 32.846715328467155
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mhr-eng)
      type: mteb/tatoeba-bitext-mining
      config: mhr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 8.0
    - type: f1
      value: 6.07664116764988
    - type: precision
      value: 5.544177607179943
    - type: recall
      value: 8.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (dan-eng)
      type: mteb/tatoeba-bitext-mining
      config: dan-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.6
    - type: f1
      value: 84.38555555555554
    - type: precision
      value: 82.91583333333334
    - type: recall
      value: 87.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ell-eng)
      type: mteb/tatoeba-bitext-mining
      config: ell-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.5
    - type: f1
      value: 84.08333333333331
    - type: precision
      value: 82.47333333333333
    - type: recall
      value: 87.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (amh-eng)
      type: mteb/tatoeba-bitext-mining
      config: amh-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 80.95238095238095
    - type: f1
      value: 76.13095238095238
    - type: precision
      value: 74.05753968253967
    - type: recall
      value: 80.95238095238095
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pam-eng)
      type: mteb/tatoeba-bitext-mining
      config: pam-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 8.799999999999999
    - type: f1
      value: 6.971422975172975
    - type: precision
      value: 6.557814916172301
    - type: recall
      value: 8.799999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hsb-eng)
      type: mteb/tatoeba-bitext-mining
      config: hsb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 44.099378881987576
    - type: f1
      value: 37.01649742022413
    - type: precision
      value: 34.69420618488942
    - type: recall
      value: 44.099378881987576
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (srp-eng)
      type: mteb/tatoeba-bitext-mining
      config: srp-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.3
    - type: f1
      value: 80.32666666666667
    - type: precision
      value: 78.60666666666665
    - type: recall
      value: 84.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (epo-eng)
      type: mteb/tatoeba-bitext-mining
      config: epo-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.5
    - type: f1
      value: 90.49666666666666
    - type: precision
      value: 89.56666666666668
    - type: recall
      value: 92.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kzj-eng)
      type: mteb/tatoeba-bitext-mining
      config: kzj-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 10.0
    - type: f1
      value: 8.268423529875141
    - type: precision
      value: 7.878118605532398
    - type: recall
      value: 10.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (awa-eng)
      type: mteb/tatoeba-bitext-mining
      config: awa-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 79.22077922077922
    - type: f1
      value: 74.27128427128426
    - type: precision
      value: 72.28715728715729
    - type: recall
      value: 79.22077922077922
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fao-eng)
      type: mteb/tatoeba-bitext-mining
      config: fao-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 65.64885496183206
    - type: f1
      value: 58.87495456197747
    - type: precision
      value: 55.992366412213734
    - type: recall
      value: 65.64885496183206
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mal-eng)
      type: mteb/tatoeba-bitext-mining
      config: mal-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.06986899563319
    - type: f1
      value: 94.78408539543909
    - type: precision
      value: 94.15332362930616
    - type: recall
      value: 96.06986899563319
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ile-eng)
      type: mteb/tatoeba-bitext-mining
      config: ile-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.2
    - type: f1
      value: 71.72571428571428
    - type: precision
      value: 69.41000000000001
    - type: recall
      value: 77.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bos-eng)
      type: mteb/tatoeba-bitext-mining
      config: bos-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.4406779661017
    - type: f1
      value: 83.2391713747646
    - type: precision
      value: 81.74199623352166
    - type: recall
      value: 86.4406779661017
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cor-eng)
      type: mteb/tatoeba-bitext-mining
      config: cor-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 8.4
    - type: f1
      value: 6.017828743398003
    - type: precision
      value: 5.4829865484756795
    - type: recall
      value: 8.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cat-eng)
      type: mteb/tatoeba-bitext-mining
      config: cat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 83.5
    - type: f1
      value: 79.74833333333333
    - type: precision
      value: 78.04837662337664
    - type: recall
      value: 83.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (eus-eng)
      type: mteb/tatoeba-bitext-mining
      config: eus-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 60.4
    - type: f1
      value: 54.467301587301584
    - type: precision
      value: 52.23242424242424
    - type: recall
      value: 60.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (yue-eng)
      type: mteb/tatoeba-bitext-mining
      config: yue-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 74.9
    - type: f1
      value: 69.68699134199134
    - type: precision
      value: 67.59873015873016
    - type: recall
      value: 74.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (swe-eng)
      type: mteb/tatoeba-bitext-mining
      config: swe-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.0
    - type: f1
      value: 84.9652380952381
    - type: precision
      value: 83.66166666666666
    - type: recall
      value: 88.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (dtp-eng)
      type: mteb/tatoeba-bitext-mining
      config: dtp-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 9.1
    - type: f1
      value: 7.681244588744588
    - type: precision
      value: 7.370043290043291
    - type: recall
      value: 9.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kat-eng)
      type: mteb/tatoeba-bitext-mining
      config: kat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 80.9651474530831
    - type: f1
      value: 76.84220605132133
    - type: precision
      value: 75.19606398962966
    - type: recall
      value: 80.9651474530831
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (jpn-eng)
      type: mteb/tatoeba-bitext-mining
      config: jpn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.9
    - type: f1
      value: 83.705
    - type: precision
      value: 82.3120634920635
    - type: recall
      value: 86.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (csb-eng)
      type: mteb/tatoeba-bitext-mining
      config: csb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 29.64426877470356
    - type: f1
      value: 23.98763072676116
    - type: precision
      value: 22.506399397703746
    - type: recall
      value: 29.64426877470356
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (xho-eng)
      type: mteb/tatoeba-bitext-mining
      config: xho-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 70.4225352112676
    - type: f1
      value: 62.84037558685445
    - type: precision
      value: 59.56572769953053
    - type: recall
      value: 70.4225352112676
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (orv-eng)
      type: mteb/tatoeba-bitext-mining
      config: orv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 19.64071856287425
    - type: f1
      value: 15.125271011207756
    - type: precision
      value: 13.865019261197494
    - type: recall
      value: 19.64071856287425
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ind-eng)
      type: mteb/tatoeba-bitext-mining
      config: ind-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.2
    - type: f1
      value: 87.80666666666666
    - type: precision
      value: 86.70833333333331
    - type: recall
      value: 90.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tuk-eng)
      type: mteb/tatoeba-bitext-mining
      config: tuk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 23.15270935960591
    - type: f1
      value: 18.407224958949097
    - type: precision
      value: 16.982385430661292
    - type: recall
      value: 23.15270935960591
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (max-eng)
      type: mteb/tatoeba-bitext-mining
      config: max-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 55.98591549295775
    - type: f1
      value: 49.94718309859154
    - type: precision
      value: 47.77864154624717
    - type: recall
      value: 55.98591549295775
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (swh-eng)
      type: mteb/tatoeba-bitext-mining
      config: swh-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 73.07692307692307
    - type: f1
      value: 66.74358974358974
    - type: precision
      value: 64.06837606837607
    - type: recall
      value: 73.07692307692307
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hin-eng)
      type: mteb/tatoeba-bitext-mining
      config: hin-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.89999999999999
    - type: f1
      value: 93.25
    - type: precision
      value: 92.43333333333332
    - type: recall
      value: 94.89999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (dsb-eng)
      type: mteb/tatoeba-bitext-mining
      config: dsb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 37.78705636743215
    - type: f1
      value: 31.63899658680452
    - type: precision
      value: 29.72264397629742
    - type: recall
      value: 37.78705636743215
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ber-eng)
      type: mteb/tatoeba-bitext-mining
      config: ber-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 21.6
    - type: f1
      value: 16.91697302697303
    - type: precision
      value: 15.71225147075147
    - type: recall
      value: 21.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tam-eng)
      type: mteb/tatoeba-bitext-mining
      config: tam-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.01628664495115
    - type: f1
      value: 81.38514037536838
    - type: precision
      value: 79.83170466883823
    - type: recall
      value: 85.01628664495115
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (slk-eng)
      type: mteb/tatoeba-bitext-mining
      config: slk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 83.39999999999999
    - type: f1
      value: 79.96380952380952
    - type: precision
      value: 78.48333333333333
    - type: recall
      value: 83.39999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tgl-eng)
      type: mteb/tatoeba-bitext-mining
      config: tgl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 83.2
    - type: f1
      value: 79.26190476190476
    - type: precision
      value: 77.58833333333334
    - type: recall
      value: 83.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ast-eng)
      type: mteb/tatoeba-bitext-mining
      config: ast-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 75.59055118110236
    - type: f1
      value: 71.66854143232096
    - type: precision
      value: 70.30183727034121
    - type: recall
      value: 75.59055118110236
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mkd-eng)
      type: mteb/tatoeba-bitext-mining
      config: mkd-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 65.5
    - type: f1
      value: 59.26095238095238
    - type: precision
      value: 56.81909090909092
    - type: recall
      value: 65.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (khm-eng)
      type: mteb/tatoeba-bitext-mining
      config: khm-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 55.26315789473685
    - type: f1
      value: 47.986523325858506
    - type: precision
      value: 45.33950006595436
    - type: recall
      value: 55.26315789473685
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ces-eng)
      type: mteb/tatoeba-bitext-mining
      config: ces-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 82.89999999999999
    - type: f1
      value: 78.835
    - type: precision
      value: 77.04761904761905
    - type: recall
      value: 82.89999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tzl-eng)
      type: mteb/tatoeba-bitext-mining
      config: tzl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 43.269230769230774
    - type: f1
      value: 36.20421245421245
    - type: precision
      value: 33.57371794871795
    - type: recall
      value: 43.269230769230774
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (urd-eng)
      type: mteb/tatoeba-bitext-mining
      config: urd-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.0
    - type: f1
      value: 84.70666666666666
    - type: precision
      value: 83.23166666666665
    - type: recall
      value: 88.0
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ara-eng)
      type: mteb/tatoeba-bitext-mining
      config: ara-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.4
    - type: f1
      value: 72.54666666666667
    - type: precision
      value: 70.54318181818181
    - type: recall
      value: 77.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kor-eng)
      type: mteb/tatoeba-bitext-mining
      config: kor-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 78.60000000000001
    - type: f1
      value: 74.1588888888889
    - type: precision
      value: 72.30250000000001
    - type: recall
      value: 78.60000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (yid-eng)
      type: mteb/tatoeba-bitext-mining
      config: yid-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 72.40566037735849
    - type: f1
      value: 66.82587328813744
    - type: precision
      value: 64.75039308176099
    - type: recall
      value: 72.40566037735849
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fin-eng)
      type: mteb/tatoeba-bitext-mining
      config: fin-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 73.8
    - type: f1
      value: 68.56357142857144
    - type: precision
      value: 66.3178822055138
    - type: recall
      value: 73.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tha-eng)
      type: mteb/tatoeba-bitext-mining
      config: tha-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.78832116788321
    - type: f1
      value: 89.3552311435523
    - type: precision
      value: 88.20559610705597
    - type: recall
      value: 91.78832116788321
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (wuu-eng)
      type: mteb/tatoeba-bitext-mining
      config: wuu-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 74.3
    - type: f1
      value: 69.05085581085581
    - type: precision
      value: 66.955
    - type: recall
      value: 74.3
  - task:
      type: Retrieval
    dataset:
      name: MTEB Touche2020
      type: webis-touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 2.896
    - type: map_at_10
      value: 8.993
    - type: map_at_100
      value: 14.133999999999999
    - type: map_at_1000
      value: 15.668000000000001
    - type: map_at_3
      value: 5.862
    - type: map_at_5
      value: 7.17
    - type: mrr_at_1
      value: 34.694
    - type: mrr_at_10
      value: 42.931000000000004
    - type: mrr_at_100
      value: 44.81
    - type: mrr_at_1000
      value: 44.81
    - type: mrr_at_3
      value: 38.435
    - type: mrr_at_5
      value: 41.701
    - type: ndcg_at_1
      value: 31.633
    - type: ndcg_at_10
      value: 21.163
    - type: ndcg_at_100
      value: 33.306000000000004
    - type: ndcg_at_1000
      value: 45.275999999999996
    - type: ndcg_at_3
      value: 25.685999999999996
    - type: ndcg_at_5
      value: 23.732
    - type: precision_at_1
      value: 34.694
    - type: precision_at_10
      value: 17.755000000000003
    - type: precision_at_100
      value: 6.938999999999999
    - type: precision_at_1000
      value: 1.48
    - type: precision_at_3
      value: 25.85
    - type: precision_at_5
      value: 23.265
    - type: recall_at_1
      value: 2.896
    - type: recall_at_10
      value: 13.333999999999998
    - type: recall_at_100
      value: 43.517
    - type: recall_at_1000
      value: 79.836
    - type: recall_at_3
      value: 6.306000000000001
    - type: recall_at_5
      value: 8.825
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
      value: 69.3874
    - type: ap
      value: 13.829909072469423
    - type: f1
      value: 53.54534203543492
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
      value: 62.62026032823995
    - type: f1
      value: 62.85251350485221
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
      value: 33.21527881409797
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
      value: 84.97943613280086
    - type: cos_sim_ap
      value: 70.75454316885921
    - type: cos_sim_f1
      value: 65.38274012676743
    - type: cos_sim_precision
      value: 60.761214318078835
    - type: cos_sim_recall
      value: 70.76517150395777
    - type: dot_accuracy
      value: 79.0546581629612
    - type: dot_ap
      value: 47.3197121792147
    - type: dot_f1
      value: 49.20106524633821
    - type: dot_precision
      value: 42.45499808502489
    - type: dot_recall
      value: 58.49604221635884
    - type: euclidean_accuracy
      value: 85.08076533349228
    - type: euclidean_ap
      value: 70.95016106374474
    - type: euclidean_f1
      value: 65.43987900176455
    - type: euclidean_precision
      value: 62.64478764478765
    - type: euclidean_recall
      value: 68.49604221635884
    - type: manhattan_accuracy
      value: 84.93771234428085
    - type: manhattan_ap
      value: 70.63668388755362
    - type: manhattan_f1
      value: 65.23895401262398
    - type: manhattan_precision
      value: 56.946084218811485
    - type: manhattan_recall
      value: 76.35883905013192
    - type: max_accuracy
      value: 85.08076533349228
    - type: max_ap
      value: 70.95016106374474
    - type: max_f1
      value: 65.43987900176455
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
      value: 88.69096130709822
    - type: cos_sim_ap
      value: 84.82526278228542
    - type: cos_sim_f1
      value: 77.65485060585536
    - type: cos_sim_precision
      value: 75.94582658619167
    - type: cos_sim_recall
      value: 79.44256236526024
    - type: dot_accuracy
      value: 80.97954748321496
    - type: dot_ap
      value: 64.81642914145866
    - type: dot_f1
      value: 60.631996987229975
    - type: dot_precision
      value: 54.5897293631712
    - type: dot_recall
      value: 68.17831844779796
    - type: euclidean_accuracy
      value: 88.6987231730508
    - type: euclidean_ap
      value: 84.80003825477253
    - type: euclidean_f1
      value: 77.67194179854496
    - type: euclidean_precision
      value: 75.7128235122094
    - type: euclidean_recall
      value: 79.73514012935017
    - type: manhattan_accuracy
      value: 88.62692591298949
    - type: manhattan_ap
      value: 84.80451408255276
    - type: manhattan_f1
      value: 77.69888949572183
    - type: manhattan_precision
      value: 73.70311528631622
    - type: manhattan_recall
      value: 82.15275639051433
    - type: max_accuracy
      value: 88.6987231730508
    - type: max_ap
      value: 84.82526278228542
    - type: max_f1
      value: 77.69888949572183
---

## Multilingual-E5-small

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

This model has 12 layers and the embedding size is 384.

## Usage

Below is an example to encode queries and passages from the MS-MARCO passage ranking dataset.

```python
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: 南瓜的家常做法',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

## Supported Languages

This model is initialized from [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)
and continually trained on a mixture of multilingual datasets.
It supports 100 languages from xlm-roberta,
but low-resource languages may see performance degradation.

## Training Details

**Initialization**: [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)

**First stage**: contrastive pre-training with weak supervision

| Dataset                                                                                                | Weak supervision                      | # of text pairs |
|--------------------------------------------------------------------------------------------------------|---------------------------------------|-----------------|
| Filtered [mC4](https://huggingface.co/datasets/mc4)                                                    | (title, page content)                 | 1B              |
| [CC News](https://huggingface.co/datasets/intfloat/multilingual_cc_news)                               | (title, news content)                 | 400M            |
| [NLLB](https://huggingface.co/datasets/allenai/nllb)                                                   | translation pairs                     | 2.4B            |
| [Wikipedia](https://huggingface.co/datasets/intfloat/wikipedia)                                        | (hierarchical section title, passage) | 150M            |
| Filtered [Reddit](https://www.reddit.com/)                                                             | (comment, response)                   | 800M            |
| [S2ORC](https://github.com/allenai/s2orc)                                                              | (title, abstract) and citation pairs  | 100M            |
| [Stackexchange](https://stackexchange.com/)                                                            | (question, answer)                    | 50M             |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)                                                  | (input prompt, response)              | 80M             |
| [Miscellaneous unsupervised SBERT data](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | -                                     | 10M             |

**Second stage**: supervised fine-tuning

| Dataset                                                                                | Language     | # of text pairs |
|----------------------------------------------------------------------------------------|--------------|-----------------|
| [MS MARCO](https://microsoft.github.io/msmarco/)                                       | English      | 500k            |
| [NQ](https://github.com/facebookresearch/DPR)                                          | English      | 70k             |
| [Trivia QA](https://github.com/facebookresearch/DPR)                                   | English      | 60k             |
| [NLI from SimCSE](https://github.com/princeton-nlp/SimCSE)                             | English      | <300k           |
| [ELI5](https://huggingface.co/datasets/eli5)                                           | English      | 500k            |
| [DuReader Retrieval](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval) | Chinese      | 86k             |
| [KILT Fever](https://huggingface.co/datasets/kilt_tasks)                               | English      | 70k             |
| [KILT HotpotQA](https://huggingface.co/datasets/kilt_tasks)                            | English      | 70k             |
| [SQuAD](https://huggingface.co/datasets/squad)                                         | English      | 87k             |
| [Quora](https://huggingface.co/datasets/quora)                                         | English      | 150k            |
| [Mr. TyDi](https://huggingface.co/datasets/castorini/mr-tydi)                                                                           | 11 languages | 50k             |
| [MIRACL](https://huggingface.co/datasets/miracl/miracl)                                                                             | 16 languages | 40k             |

For all labeled datasets, we only use its training set for fine-tuning.

For other training details, please refer to our paper at [https://arxiv.org/pdf/2212.03533.pdf](https://arxiv.org/pdf/2212.03533.pdf).

## Benchmark Results on [Mr. TyDi](https://arxiv.org/abs/2108.08787)

| Model                 | Avg MRR@10 |       | ar   | bn | en | fi | id | ja | ko | ru | sw   | te | th |
|-----------------------|------------|-------|------| --- | --- | --- | --- | --- | --- | --- |------| --- | --- |
| BM25                  | 33.3       | | 36.7 | 41.3 | 15.1 | 28.8 | 38.2 | 21.7 | 28.1 | 32.9 | 39.6 | 42.4 | 41.7 |
| mDPR                  | 16.7       | | 26.0 | 25.8  | 16.2 | 11.3 | 14.6 | 18.1 | 21.9 | 18.5 | 7.3 | 10.6 | 13.5 |
| BM25 + mDPR           | 41.7       | | 49.1 | 53.5 | 28.4 | 36.5 | 45.5 | 35.5 | 36.2 | 42.7 | 40.5 | 42.0 | 49.2 |
|                       |            |
| multilingual-e5-small | 64.4       | | 71.5 | 66.3 | 54.5 | 57.7 | 63.2 | 55.4 | 54.3 | 60.8 | 65.4 | 89.1 | 70.1 |
| multilingual-e5-base  | 65.9       | | 72.3 | 65.0 | 58.5 | 60.8 | 64.9 | 56.6 | 55.8 | 62.7 | 69.0 | 86.6 | 72.7 |
| multilingual-e5-large | **70.5**   | | 77.5 | 73.2 | 60.8 | 66.8 | 68.5 | 62.5 | 61.6 | 65.8 | 72.7 | 90.2 | 76.2 |

## MTEB Benchmark Evaluation

Check out [unilm/e5](https://github.com/microsoft/unilm/tree/master/e5) to reproduce evaluation results 
on the [BEIR](https://arxiv.org/abs/2104.08663) and [MTEB benchmark](https://arxiv.org/abs/2210.07316).

## Support for Sentence Transformers

Below is an example for usage with sentence_transformers.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-small')
input_texts = [
    'query: how much protein should a female eat',
    'query: 南瓜的家常做法',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 i     s 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or traini     ng for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮     ,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,     放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油     锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀      6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]
embeddings = model.encode(input_texts, normalize_embeddings=True)
```

Package requirements

`pip install sentence_transformers~=2.2.2`

Contributors: [michaelfeil](https://huggingface.co/michaelfeil)

## FAQ

**1. Do I need to add the prefix "query: " and "passage: " to input texts?**

Yes, this is how the model is trained, otherwise you will see a performance degradation.

Here are some rules of thumb:
- Use "query: " and "passage: " correspondingly for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.

- Use "query: " prefix for symmetric tasks such as semantic similarity, bitext mining, paraphrase retrieval.

- Use "query: " prefix if you want to use embeddings as features, such as linear probing classification, clustering.

**2. Why are my reproduced results slightly different from reported in the model card?**

Different versions of `transformers` and `pytorch` could cause negligible but non-zero performance differences.

**3. Why does the cosine similarity scores distribute around 0.7 to 1.0?**

This is a known and expected behavior as we use a low temperature 0.01 for InfoNCE contrastive loss. 

For text embedding tasks like text retrieval or semantic similarity, 
what matters is the relative order of the scores instead of the absolute values, 
so this should not be an issue.

## Citation

If you find our paper or models helpful, please consider cite as follows:

```
@article{wang2022text,
  title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2212.03533},
  year={2022}
}
```

## Limitations

Long texts will be truncated to at most 512 tokens.

