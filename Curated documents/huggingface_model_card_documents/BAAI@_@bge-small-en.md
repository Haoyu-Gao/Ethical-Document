---
language:
- en
license: mit
tags:
- mteb
- sentence transformers
model-index:
- name: bge-small-en
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
      value: 74.34328358208955
    - type: ap
      value: 37.59947775195661
    - type: f1
      value: 68.548415491933
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
      value: 93.04527499999999
    - type: ap
      value: 89.60696356772135
    - type: f1
      value: 93.03361469382438
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
      value: 46.08
    - type: f1
      value: 45.66249835363254
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
      value: 35.205999999999996
    - type: map_at_10
      value: 50.782000000000004
    - type: map_at_100
      value: 51.547
    - type: map_at_1000
      value: 51.554
    - type: map_at_3
      value: 46.515
    - type: map_at_5
      value: 49.296
    - type: mrr_at_1
      value: 35.632999999999996
    - type: mrr_at_10
      value: 50.958999999999996
    - type: mrr_at_100
      value: 51.724000000000004
    - type: mrr_at_1000
      value: 51.731
    - type: mrr_at_3
      value: 46.669
    - type: mrr_at_5
      value: 49.439
    - type: ndcg_at_1
      value: 35.205999999999996
    - type: ndcg_at_10
      value: 58.835
    - type: ndcg_at_100
      value: 62.095
    - type: ndcg_at_1000
      value: 62.255
    - type: ndcg_at_3
      value: 50.255
    - type: ndcg_at_5
      value: 55.296
    - type: precision_at_1
      value: 35.205999999999996
    - type: precision_at_10
      value: 8.421
    - type: precision_at_100
      value: 0.984
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 20.365
    - type: precision_at_5
      value: 14.680000000000001
    - type: recall_at_1
      value: 35.205999999999996
    - type: recall_at_10
      value: 84.211
    - type: recall_at_100
      value: 98.43499999999999
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 61.095
    - type: recall_at_5
      value: 73.4
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
      value: 47.52644476278646
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
      value: 39.973045724188964
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
      value: 62.28285314871488
    - type: mrr
      value: 74.52743701358659
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
      value: 80.09041909160327
    - type: cos_sim_spearman
      value: 79.96266537706944
    - type: euclidean_pearson
      value: 79.50774978162241
    - type: euclidean_spearman
      value: 79.9144715078551
    - type: manhattan_pearson
      value: 79.2062139879302
    - type: manhattan_spearman
      value: 79.35000081468212
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
      value: 85.31493506493506
    - type: f1
      value: 85.2704557977762
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
      value: 39.6837242810816
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
      value: 35.38881249555897
  - task:
      type: Retrieval
    dataset:
      name: MTEB CQADupstackAndroidRetrieval
      type: BeIR/cqadupstack
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.884999999999998
    - type: map_at_10
      value: 39.574
    - type: map_at_100
      value: 40.993
    - type: map_at_1000
      value: 41.129
    - type: map_at_3
      value: 36.089
    - type: map_at_5
      value: 38.191
    - type: mrr_at_1
      value: 34.477999999999994
    - type: mrr_at_10
      value: 45.411
    - type: mrr_at_100
      value: 46.089999999999996
    - type: mrr_at_1000
      value: 46.147
    - type: mrr_at_3
      value: 42.346000000000004
    - type: mrr_at_5
      value: 44.292
    - type: ndcg_at_1
      value: 34.477999999999994
    - type: ndcg_at_10
      value: 46.123999999999995
    - type: ndcg_at_100
      value: 51.349999999999994
    - type: ndcg_at_1000
      value: 53.578
    - type: ndcg_at_3
      value: 40.824
    - type: ndcg_at_5
      value: 43.571
    - type: precision_at_1
      value: 34.477999999999994
    - type: precision_at_10
      value: 8.841000000000001
    - type: precision_at_100
      value: 1.4460000000000002
    - type: precision_at_1000
      value: 0.192
    - type: precision_at_3
      value: 19.742
    - type: precision_at_5
      value: 14.421000000000001
    - type: recall_at_1
      value: 27.884999999999998
    - type: recall_at_10
      value: 59.087
    - type: recall_at_100
      value: 80.609
    - type: recall_at_1000
      value: 95.054
    - type: recall_at_3
      value: 44.082
    - type: recall_at_5
      value: 51.593999999999994
    - type: map_at_1
      value: 30.639
    - type: map_at_10
      value: 40.047
    - type: map_at_100
      value: 41.302
    - type: map_at_1000
      value: 41.425
    - type: map_at_3
      value: 37.406
    - type: map_at_5
      value: 38.934000000000005
    - type: mrr_at_1
      value: 37.707
    - type: mrr_at_10
      value: 46.082
    - type: mrr_at_100
      value: 46.745
    - type: mrr_at_1000
      value: 46.786
    - type: mrr_at_3
      value: 43.980999999999995
    - type: mrr_at_5
      value: 45.287
    - type: ndcg_at_1
      value: 37.707
    - type: ndcg_at_10
      value: 45.525
    - type: ndcg_at_100
      value: 49.976
    - type: ndcg_at_1000
      value: 51.94499999999999
    - type: ndcg_at_3
      value: 41.704
    - type: ndcg_at_5
      value: 43.596000000000004
    - type: precision_at_1
      value: 37.707
    - type: precision_at_10
      value: 8.465
    - type: precision_at_100
      value: 1.375
    - type: precision_at_1000
      value: 0.183
    - type: precision_at_3
      value: 19.979
    - type: precision_at_5
      value: 14.115
    - type: recall_at_1
      value: 30.639
    - type: recall_at_10
      value: 54.775
    - type: recall_at_100
      value: 73.678
    - type: recall_at_1000
      value: 86.142
    - type: recall_at_3
      value: 43.230000000000004
    - type: recall_at_5
      value: 48.622
    - type: map_at_1
      value: 38.038
    - type: map_at_10
      value: 49.922
    - type: map_at_100
      value: 51.032
    - type: map_at_1000
      value: 51.085
    - type: map_at_3
      value: 46.664
    - type: map_at_5
      value: 48.588
    - type: mrr_at_1
      value: 43.95
    - type: mrr_at_10
      value: 53.566
    - type: mrr_at_100
      value: 54.318999999999996
    - type: mrr_at_1000
      value: 54.348
    - type: mrr_at_3
      value: 51.066
    - type: mrr_at_5
      value: 52.649
    - type: ndcg_at_1
      value: 43.95
    - type: ndcg_at_10
      value: 55.676
    - type: ndcg_at_100
      value: 60.126000000000005
    - type: ndcg_at_1000
      value: 61.208
    - type: ndcg_at_3
      value: 50.20400000000001
    - type: ndcg_at_5
      value: 53.038
    - type: precision_at_1
      value: 43.95
    - type: precision_at_10
      value: 8.953
    - type: precision_at_100
      value: 1.2109999999999999
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 22.256999999999998
    - type: precision_at_5
      value: 15.524
    - type: recall_at_1
      value: 38.038
    - type: recall_at_10
      value: 69.15
    - type: recall_at_100
      value: 88.31599999999999
    - type: recall_at_1000
      value: 95.993
    - type: recall_at_3
      value: 54.663
    - type: recall_at_5
      value: 61.373
    - type: map_at_1
      value: 24.872
    - type: map_at_10
      value: 32.912
    - type: map_at_100
      value: 33.972
    - type: map_at_1000
      value: 34.046
    - type: map_at_3
      value: 30.361
    - type: map_at_5
      value: 31.704
    - type: mrr_at_1
      value: 26.779999999999998
    - type: mrr_at_10
      value: 34.812
    - type: mrr_at_100
      value: 35.754999999999995
    - type: mrr_at_1000
      value: 35.809000000000005
    - type: mrr_at_3
      value: 32.335
    - type: mrr_at_5
      value: 33.64
    - type: ndcg_at_1
      value: 26.779999999999998
    - type: ndcg_at_10
      value: 37.623
    - type: ndcg_at_100
      value: 42.924
    - type: ndcg_at_1000
      value: 44.856
    - type: ndcg_at_3
      value: 32.574
    - type: ndcg_at_5
      value: 34.842
    - type: precision_at_1
      value: 26.779999999999998
    - type: precision_at_10
      value: 5.729
    - type: precision_at_100
      value: 0.886
    - type: precision_at_1000
      value: 0.109
    - type: precision_at_3
      value: 13.559
    - type: precision_at_5
      value: 9.469
    - type: recall_at_1
      value: 24.872
    - type: recall_at_10
      value: 50.400999999999996
    - type: recall_at_100
      value: 74.954
    - type: recall_at_1000
      value: 89.56
    - type: recall_at_3
      value: 36.726
    - type: recall_at_5
      value: 42.138999999999996
    - type: map_at_1
      value: 16.803
    - type: map_at_10
      value: 24.348
    - type: map_at_100
      value: 25.56
    - type: map_at_1000
      value: 25.668000000000003
    - type: map_at_3
      value: 21.811
    - type: map_at_5
      value: 23.287
    - type: mrr_at_1
      value: 20.771
    - type: mrr_at_10
      value: 28.961
    - type: mrr_at_100
      value: 29.979
    - type: mrr_at_1000
      value: 30.046
    - type: mrr_at_3
      value: 26.555
    - type: mrr_at_5
      value: 28.060000000000002
    - type: ndcg_at_1
      value: 20.771
    - type: ndcg_at_10
      value: 29.335
    - type: ndcg_at_100
      value: 35.188
    - type: ndcg_at_1000
      value: 37.812
    - type: ndcg_at_3
      value: 24.83
    - type: ndcg_at_5
      value: 27.119
    - type: precision_at_1
      value: 20.771
    - type: precision_at_10
      value: 5.4350000000000005
    - type: precision_at_100
      value: 0.9480000000000001
    - type: precision_at_1000
      value: 0.13
    - type: precision_at_3
      value: 11.982
    - type: precision_at_5
      value: 8.831
    - type: recall_at_1
      value: 16.803
    - type: recall_at_10
      value: 40.039
    - type: recall_at_100
      value: 65.83200000000001
    - type: recall_at_1000
      value: 84.478
    - type: recall_at_3
      value: 27.682000000000002
    - type: recall_at_5
      value: 33.535
    - type: map_at_1
      value: 28.345
    - type: map_at_10
      value: 37.757000000000005
    - type: map_at_100
      value: 39.141
    - type: map_at_1000
      value: 39.262
    - type: map_at_3
      value: 35.183
    - type: map_at_5
      value: 36.592
    - type: mrr_at_1
      value: 34.649
    - type: mrr_at_10
      value: 43.586999999999996
    - type: mrr_at_100
      value: 44.481
    - type: mrr_at_1000
      value: 44.542
    - type: mrr_at_3
      value: 41.29
    - type: mrr_at_5
      value: 42.642
    - type: ndcg_at_1
      value: 34.649
    - type: ndcg_at_10
      value: 43.161
    - type: ndcg_at_100
      value: 48.734
    - type: ndcg_at_1000
      value: 51.046
    - type: ndcg_at_3
      value: 39.118
    - type: ndcg_at_5
      value: 41.022
    - type: precision_at_1
      value: 34.649
    - type: precision_at_10
      value: 7.603
    - type: precision_at_100
      value: 1.209
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 18.319
    - type: precision_at_5
      value: 12.839
    - type: recall_at_1
      value: 28.345
    - type: recall_at_10
      value: 53.367
    - type: recall_at_100
      value: 76.453
    - type: recall_at_1000
      value: 91.82000000000001
    - type: recall_at_3
      value: 41.636
    - type: recall_at_5
      value: 46.760000000000005
    - type: map_at_1
      value: 22.419
    - type: map_at_10
      value: 31.716
    - type: map_at_100
      value: 33.152
    - type: map_at_1000
      value: 33.267
    - type: map_at_3
      value: 28.74
    - type: map_at_5
      value: 30.48
    - type: mrr_at_1
      value: 28.310999999999996
    - type: mrr_at_10
      value: 37.039
    - type: mrr_at_100
      value: 38.09
    - type: mrr_at_1000
      value: 38.145
    - type: mrr_at_3
      value: 34.437
    - type: mrr_at_5
      value: 36.024
    - type: ndcg_at_1
      value: 28.310999999999996
    - type: ndcg_at_10
      value: 37.41
    - type: ndcg_at_100
      value: 43.647999999999996
    - type: ndcg_at_1000
      value: 46.007
    - type: ndcg_at_3
      value: 32.509
    - type: ndcg_at_5
      value: 34.943999999999996
    - type: precision_at_1
      value: 28.310999999999996
    - type: precision_at_10
      value: 6.963
    - type: precision_at_100
      value: 1.1860000000000002
    - type: precision_at_1000
      value: 0.154
    - type: precision_at_3
      value: 15.867999999999999
    - type: precision_at_5
      value: 11.507000000000001
    - type: recall_at_1
      value: 22.419
    - type: recall_at_10
      value: 49.28
    - type: recall_at_100
      value: 75.802
    - type: recall_at_1000
      value: 92.032
    - type: recall_at_3
      value: 35.399
    - type: recall_at_5
      value: 42.027
    - type: map_at_1
      value: 24.669249999999998
    - type: map_at_10
      value: 33.332583333333325
    - type: map_at_100
      value: 34.557833333333335
    - type: map_at_1000
      value: 34.67141666666666
    - type: map_at_3
      value: 30.663166666666662
    - type: map_at_5
      value: 32.14883333333333
    - type: mrr_at_1
      value: 29.193833333333334
    - type: mrr_at_10
      value: 37.47625
    - type: mrr_at_100
      value: 38.3545
    - type: mrr_at_1000
      value: 38.413166666666676
    - type: mrr_at_3
      value: 35.06741666666667
    - type: mrr_at_5
      value: 36.450666666666656
    - type: ndcg_at_1
      value: 29.193833333333334
    - type: ndcg_at_10
      value: 38.505416666666676
    - type: ndcg_at_100
      value: 43.81125
    - type: ndcg_at_1000
      value: 46.09558333333333
    - type: ndcg_at_3
      value: 33.90916666666667
    - type: ndcg_at_5
      value: 36.07666666666666
    - type: precision_at_1
      value: 29.193833333333334
    - type: precision_at_10
      value: 6.7251666666666665
    - type: precision_at_100
      value: 1.1058333333333332
    - type: precision_at_1000
      value: 0.14833333333333332
    - type: precision_at_3
      value: 15.554166666666665
    - type: precision_at_5
      value: 11.079250000000002
    - type: recall_at_1
      value: 24.669249999999998
    - type: recall_at_10
      value: 49.75583333333332
    - type: recall_at_100
      value: 73.06908333333332
    - type: recall_at_1000
      value: 88.91316666666667
    - type: recall_at_3
      value: 36.913250000000005
    - type: recall_at_5
      value: 42.48641666666666
    - type: map_at_1
      value: 24.044999999999998
    - type: map_at_10
      value: 30.349999999999998
    - type: map_at_100
      value: 31.273
    - type: map_at_1000
      value: 31.362000000000002
    - type: map_at_3
      value: 28.508
    - type: map_at_5
      value: 29.369
    - type: mrr_at_1
      value: 26.994
    - type: mrr_at_10
      value: 33.12
    - type: mrr_at_100
      value: 33.904
    - type: mrr_at_1000
      value: 33.967000000000006
    - type: mrr_at_3
      value: 31.365
    - type: mrr_at_5
      value: 32.124
    - type: ndcg_at_1
      value: 26.994
    - type: ndcg_at_10
      value: 34.214
    - type: ndcg_at_100
      value: 38.681
    - type: ndcg_at_1000
      value: 40.926
    - type: ndcg_at_3
      value: 30.725
    - type: ndcg_at_5
      value: 31.967000000000002
    - type: precision_at_1
      value: 26.994
    - type: precision_at_10
      value: 5.215
    - type: precision_at_100
      value: 0.807
    - type: precision_at_1000
      value: 0.108
    - type: precision_at_3
      value: 12.986
    - type: precision_at_5
      value: 8.712
    - type: recall_at_1
      value: 24.044999999999998
    - type: recall_at_10
      value: 43.456
    - type: recall_at_100
      value: 63.675000000000004
    - type: recall_at_1000
      value: 80.05499999999999
    - type: recall_at_3
      value: 33.561
    - type: recall_at_5
      value: 36.767
    - type: map_at_1
      value: 15.672
    - type: map_at_10
      value: 22.641
    - type: map_at_100
      value: 23.75
    - type: map_at_1000
      value: 23.877000000000002
    - type: map_at_3
      value: 20.219
    - type: map_at_5
      value: 21.648
    - type: mrr_at_1
      value: 18.823
    - type: mrr_at_10
      value: 26.101999999999997
    - type: mrr_at_100
      value: 27.038
    - type: mrr_at_1000
      value: 27.118
    - type: mrr_at_3
      value: 23.669
    - type: mrr_at_5
      value: 25.173000000000002
    - type: ndcg_at_1
      value: 18.823
    - type: ndcg_at_10
      value: 27.176000000000002
    - type: ndcg_at_100
      value: 32.42
    - type: ndcg_at_1000
      value: 35.413
    - type: ndcg_at_3
      value: 22.756999999999998
    - type: ndcg_at_5
      value: 25.032
    - type: precision_at_1
      value: 18.823
    - type: precision_at_10
      value: 5.034000000000001
    - type: precision_at_100
      value: 0.895
    - type: precision_at_1000
      value: 0.132
    - type: precision_at_3
      value: 10.771
    - type: precision_at_5
      value: 8.1
    - type: recall_at_1
      value: 15.672
    - type: recall_at_10
      value: 37.296
    - type: recall_at_100
      value: 60.863
    - type: recall_at_1000
      value: 82.234
    - type: recall_at_3
      value: 25.330000000000002
    - type: recall_at_5
      value: 30.964000000000002
    - type: map_at_1
      value: 24.633
    - type: map_at_10
      value: 32.858
    - type: map_at_100
      value: 34.038000000000004
    - type: map_at_1000
      value: 34.141
    - type: map_at_3
      value: 30.209000000000003
    - type: map_at_5
      value: 31.567
    - type: mrr_at_1
      value: 28.358
    - type: mrr_at_10
      value: 36.433
    - type: mrr_at_100
      value: 37.352000000000004
    - type: mrr_at_1000
      value: 37.41
    - type: mrr_at_3
      value: 34.033
    - type: mrr_at_5
      value: 35.246
    - type: ndcg_at_1
      value: 28.358
    - type: ndcg_at_10
      value: 37.973
    - type: ndcg_at_100
      value: 43.411
    - type: ndcg_at_1000
      value: 45.747
    - type: ndcg_at_3
      value: 32.934999999999995
    - type: ndcg_at_5
      value: 35.013
    - type: precision_at_1
      value: 28.358
    - type: precision_at_10
      value: 6.418
    - type: precision_at_100
      value: 1.02
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 14.677000000000001
    - type: precision_at_5
      value: 10.335999999999999
    - type: recall_at_1
      value: 24.633
    - type: recall_at_10
      value: 50.048
    - type: recall_at_100
      value: 73.821
    - type: recall_at_1000
      value: 90.046
    - type: recall_at_3
      value: 36.284
    - type: recall_at_5
      value: 41.370000000000005
    - type: map_at_1
      value: 23.133
    - type: map_at_10
      value: 31.491999999999997
    - type: map_at_100
      value: 33.062000000000005
    - type: map_at_1000
      value: 33.256
    - type: map_at_3
      value: 28.886
    - type: map_at_5
      value: 30.262
    - type: mrr_at_1
      value: 28.063
    - type: mrr_at_10
      value: 36.144
    - type: mrr_at_100
      value: 37.14
    - type: mrr_at_1000
      value: 37.191
    - type: mrr_at_3
      value: 33.762
    - type: mrr_at_5
      value: 34.997
    - type: ndcg_at_1
      value: 28.063
    - type: ndcg_at_10
      value: 36.951
    - type: ndcg_at_100
      value: 43.287
    - type: ndcg_at_1000
      value: 45.777
    - type: ndcg_at_3
      value: 32.786
    - type: ndcg_at_5
      value: 34.65
    - type: precision_at_1
      value: 28.063
    - type: precision_at_10
      value: 7.055
    - type: precision_at_100
      value: 1.476
    - type: precision_at_1000
      value: 0.22899999999999998
    - type: precision_at_3
      value: 15.481
    - type: precision_at_5
      value: 11.186
    - type: recall_at_1
      value: 23.133
    - type: recall_at_10
      value: 47.285
    - type: recall_at_100
      value: 76.176
    - type: recall_at_1000
      value: 92.176
    - type: recall_at_3
      value: 35.223
    - type: recall_at_5
      value: 40.142
    - type: map_at_1
      value: 19.547
    - type: map_at_10
      value: 26.374
    - type: map_at_100
      value: 27.419
    - type: map_at_1000
      value: 27.539
    - type: map_at_3
      value: 23.882
    - type: map_at_5
      value: 25.163999999999998
    - type: mrr_at_1
      value: 21.442
    - type: mrr_at_10
      value: 28.458
    - type: mrr_at_100
      value: 29.360999999999997
    - type: mrr_at_1000
      value: 29.448999999999998
    - type: mrr_at_3
      value: 25.97
    - type: mrr_at_5
      value: 27.273999999999997
    - type: ndcg_at_1
      value: 21.442
    - type: ndcg_at_10
      value: 30.897000000000002
    - type: ndcg_at_100
      value: 35.99
    - type: ndcg_at_1000
      value: 38.832
    - type: ndcg_at_3
      value: 25.944
    - type: ndcg_at_5
      value: 28.126
    - type: precision_at_1
      value: 21.442
    - type: precision_at_10
      value: 4.9910000000000005
    - type: precision_at_100
      value: 0.8109999999999999
    - type: precision_at_1000
      value: 0.11800000000000001
    - type: precision_at_3
      value: 11.029
    - type: precision_at_5
      value: 7.911
    - type: recall_at_1
      value: 19.547
    - type: recall_at_10
      value: 42.886
    - type: recall_at_100
      value: 66.64999999999999
    - type: recall_at_1000
      value: 87.368
    - type: recall_at_3
      value: 29.143
    - type: recall_at_5
      value: 34.544000000000004
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
      value: 15.572
    - type: map_at_10
      value: 25.312
    - type: map_at_100
      value: 27.062
    - type: map_at_1000
      value: 27.253
    - type: map_at_3
      value: 21.601
    - type: map_at_5
      value: 23.473
    - type: mrr_at_1
      value: 34.984
    - type: mrr_at_10
      value: 46.406
    - type: mrr_at_100
      value: 47.179
    - type: mrr_at_1000
      value: 47.21
    - type: mrr_at_3
      value: 43.485
    - type: mrr_at_5
      value: 45.322
    - type: ndcg_at_1
      value: 34.984
    - type: ndcg_at_10
      value: 34.344
    - type: ndcg_at_100
      value: 41.015
    - type: ndcg_at_1000
      value: 44.366
    - type: ndcg_at_3
      value: 29.119
    - type: ndcg_at_5
      value: 30.825999999999997
    - type: precision_at_1
      value: 34.984
    - type: precision_at_10
      value: 10.358
    - type: precision_at_100
      value: 1.762
    - type: precision_at_1000
      value: 0.23900000000000002
    - type: precision_at_3
      value: 21.368000000000002
    - type: precision_at_5
      value: 15.948
    - type: recall_at_1
      value: 15.572
    - type: recall_at_10
      value: 39.367999999999995
    - type: recall_at_100
      value: 62.183
    - type: recall_at_1000
      value: 80.92200000000001
    - type: recall_at_3
      value: 26.131999999999998
    - type: recall_at_5
      value: 31.635999999999996
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
      value: 8.848
    - type: map_at_10
      value: 19.25
    - type: map_at_100
      value: 27.193
    - type: map_at_1000
      value: 28.721999999999998
    - type: map_at_3
      value: 13.968
    - type: map_at_5
      value: 16.283
    - type: mrr_at_1
      value: 68.75
    - type: mrr_at_10
      value: 76.25
    - type: mrr_at_100
      value: 76.534
    - type: mrr_at_1000
      value: 76.53999999999999
    - type: mrr_at_3
      value: 74.667
    - type: mrr_at_5
      value: 75.86699999999999
    - type: ndcg_at_1
      value: 56.00000000000001
    - type: ndcg_at_10
      value: 41.426
    - type: ndcg_at_100
      value: 45.660000000000004
    - type: ndcg_at_1000
      value: 53.02
    - type: ndcg_at_3
      value: 46.581
    - type: ndcg_at_5
      value: 43.836999999999996
    - type: precision_at_1
      value: 68.75
    - type: precision_at_10
      value: 32.800000000000004
    - type: precision_at_100
      value: 10.440000000000001
    - type: precision_at_1000
      value: 1.9980000000000002
    - type: precision_at_3
      value: 49.667
    - type: precision_at_5
      value: 42.25
    - type: recall_at_1
      value: 8.848
    - type: recall_at_10
      value: 24.467
    - type: recall_at_100
      value: 51.344
    - type: recall_at_1000
      value: 75.235
    - type: recall_at_3
      value: 15.329
    - type: recall_at_5
      value: 18.892999999999997
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
      value: 48.95
    - type: f1
      value: 43.44563593360779
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
      value: 78.036
    - type: map_at_10
      value: 85.639
    - type: map_at_100
      value: 85.815
    - type: map_at_1000
      value: 85.829
    - type: map_at_3
      value: 84.795
    - type: map_at_5
      value: 85.336
    - type: mrr_at_1
      value: 84.353
    - type: mrr_at_10
      value: 90.582
    - type: mrr_at_100
      value: 90.617
    - type: mrr_at_1000
      value: 90.617
    - type: mrr_at_3
      value: 90.132
    - type: mrr_at_5
      value: 90.447
    - type: ndcg_at_1
      value: 84.353
    - type: ndcg_at_10
      value: 89.003
    - type: ndcg_at_100
      value: 89.60000000000001
    - type: ndcg_at_1000
      value: 89.836
    - type: ndcg_at_3
      value: 87.81400000000001
    - type: ndcg_at_5
      value: 88.478
    - type: precision_at_1
      value: 84.353
    - type: precision_at_10
      value: 10.482
    - type: precision_at_100
      value: 1.099
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 33.257999999999996
    - type: precision_at_5
      value: 20.465
    - type: recall_at_1
      value: 78.036
    - type: recall_at_10
      value: 94.517
    - type: recall_at_100
      value: 96.828
    - type: recall_at_1000
      value: 98.261
    - type: recall_at_3
      value: 91.12
    - type: recall_at_5
      value: 92.946
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
      value: 20.191
    - type: map_at_10
      value: 32.369
    - type: map_at_100
      value: 34.123999999999995
    - type: map_at_1000
      value: 34.317
    - type: map_at_3
      value: 28.71
    - type: map_at_5
      value: 30.607
    - type: mrr_at_1
      value: 40.894999999999996
    - type: mrr_at_10
      value: 48.842
    - type: mrr_at_100
      value: 49.599
    - type: mrr_at_1000
      value: 49.647000000000006
    - type: mrr_at_3
      value: 46.785
    - type: mrr_at_5
      value: 47.672
    - type: ndcg_at_1
      value: 40.894999999999996
    - type: ndcg_at_10
      value: 39.872
    - type: ndcg_at_100
      value: 46.126
    - type: ndcg_at_1000
      value: 49.476
    - type: ndcg_at_3
      value: 37.153000000000006
    - type: ndcg_at_5
      value: 37.433
    - type: precision_at_1
      value: 40.894999999999996
    - type: precision_at_10
      value: 10.818
    - type: precision_at_100
      value: 1.73
    - type: precision_at_1000
      value: 0.231
    - type: precision_at_3
      value: 25.051000000000002
    - type: precision_at_5
      value: 17.531
    - type: recall_at_1
      value: 20.191
    - type: recall_at_10
      value: 45.768
    - type: recall_at_100
      value: 68.82000000000001
    - type: recall_at_1000
      value: 89.133
    - type: recall_at_3
      value: 33.296
    - type: recall_at_5
      value: 38.022
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
      value: 39.257
    - type: map_at_10
      value: 61.467000000000006
    - type: map_at_100
      value: 62.364
    - type: map_at_1000
      value: 62.424
    - type: map_at_3
      value: 58.228
    - type: map_at_5
      value: 60.283
    - type: mrr_at_1
      value: 78.515
    - type: mrr_at_10
      value: 84.191
    - type: mrr_at_100
      value: 84.378
    - type: mrr_at_1000
      value: 84.385
    - type: mrr_at_3
      value: 83.284
    - type: mrr_at_5
      value: 83.856
    - type: ndcg_at_1
      value: 78.515
    - type: ndcg_at_10
      value: 69.78999999999999
    - type: ndcg_at_100
      value: 72.886
    - type: ndcg_at_1000
      value: 74.015
    - type: ndcg_at_3
      value: 65.23
    - type: ndcg_at_5
      value: 67.80199999999999
    - type: precision_at_1
      value: 78.515
    - type: precision_at_10
      value: 14.519000000000002
    - type: precision_at_100
      value: 1.694
    - type: precision_at_1000
      value: 0.184
    - type: precision_at_3
      value: 41.702
    - type: precision_at_5
      value: 27.046999999999997
    - type: recall_at_1
      value: 39.257
    - type: recall_at_10
      value: 72.59299999999999
    - type: recall_at_100
      value: 84.679
    - type: recall_at_1000
      value: 92.12
    - type: recall_at_3
      value: 62.552
    - type: recall_at_5
      value: 67.616
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
      value: 91.5152
    - type: ap
      value: 87.64584669595709
    - type: f1
      value: 91.50605576428437
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
      value: 21.926000000000002
    - type: map_at_10
      value: 34.049
    - type: map_at_100
      value: 35.213
    - type: map_at_1000
      value: 35.265
    - type: map_at_3
      value: 30.309
    - type: map_at_5
      value: 32.407000000000004
    - type: mrr_at_1
      value: 22.55
    - type: mrr_at_10
      value: 34.657
    - type: mrr_at_100
      value: 35.760999999999996
    - type: mrr_at_1000
      value: 35.807
    - type: mrr_at_3
      value: 30.989
    - type: mrr_at_5
      value: 33.039
    - type: ndcg_at_1
      value: 22.55
    - type: ndcg_at_10
      value: 40.842
    - type: ndcg_at_100
      value: 46.436
    - type: ndcg_at_1000
      value: 47.721999999999994
    - type: ndcg_at_3
      value: 33.209
    - type: ndcg_at_5
      value: 36.943
    - type: precision_at_1
      value: 22.55
    - type: precision_at_10
      value: 6.447
    - type: precision_at_100
      value: 0.9249999999999999
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.136000000000001
    - type: precision_at_5
      value: 10.381
    - type: recall_at_1
      value: 21.926000000000002
    - type: recall_at_10
      value: 61.724999999999994
    - type: recall_at_100
      value: 87.604
    - type: recall_at_1000
      value: 97.421
    - type: recall_at_3
      value: 40.944
    - type: recall_at_5
      value: 49.915
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
      value: 93.54765161878704
    - type: f1
      value: 93.3298945415573
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
      value: 75.71591427268582
    - type: f1
      value: 59.32113870474471
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
      value: 75.83053127101547
    - type: f1
      value: 73.60757944876475
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
      value: 78.72562205783457
    - type: f1
      value: 78.63761662505502
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
      value: 33.37935633767996
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
      value: 31.55270546130387
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
      value: 30.462692753143834
    - type: mrr
      value: 31.497569753511563
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
      value: 5.646
    - type: map_at_10
      value: 12.498
    - type: map_at_100
      value: 15.486
    - type: map_at_1000
      value: 16.805999999999997
    - type: map_at_3
      value: 9.325
    - type: map_at_5
      value: 10.751
    - type: mrr_at_1
      value: 43.034
    - type: mrr_at_10
      value: 52.662
    - type: mrr_at_100
      value: 53.189
    - type: mrr_at_1000
      value: 53.25
    - type: mrr_at_3
      value: 50.929
    - type: mrr_at_5
      value: 51.92
    - type: ndcg_at_1
      value: 41.796
    - type: ndcg_at_10
      value: 33.477000000000004
    - type: ndcg_at_100
      value: 29.996000000000002
    - type: ndcg_at_1000
      value: 38.864
    - type: ndcg_at_3
      value: 38.940000000000005
    - type: ndcg_at_5
      value: 36.689
    - type: precision_at_1
      value: 43.034
    - type: precision_at_10
      value: 24.799
    - type: precision_at_100
      value: 7.432999999999999
    - type: precision_at_1000
      value: 1.9929999999999999
    - type: precision_at_3
      value: 36.842000000000006
    - type: precision_at_5
      value: 32.135999999999996
    - type: recall_at_1
      value: 5.646
    - type: recall_at_10
      value: 15.963
    - type: recall_at_100
      value: 29.492
    - type: recall_at_1000
      value: 61.711000000000006
    - type: recall_at_3
      value: 10.585
    - type: recall_at_5
      value: 12.753999999999998
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
      value: 27.602
    - type: map_at_10
      value: 41.545
    - type: map_at_100
      value: 42.644999999999996
    - type: map_at_1000
      value: 42.685
    - type: map_at_3
      value: 37.261
    - type: map_at_5
      value: 39.706
    - type: mrr_at_1
      value: 31.141000000000002
    - type: mrr_at_10
      value: 44.139
    - type: mrr_at_100
      value: 44.997
    - type: mrr_at_1000
      value: 45.025999999999996
    - type: mrr_at_3
      value: 40.503
    - type: mrr_at_5
      value: 42.64
    - type: ndcg_at_1
      value: 31.141000000000002
    - type: ndcg_at_10
      value: 48.995
    - type: ndcg_at_100
      value: 53.788000000000004
    - type: ndcg_at_1000
      value: 54.730000000000004
    - type: ndcg_at_3
      value: 40.844
    - type: ndcg_at_5
      value: 44.955
    - type: precision_at_1
      value: 31.141000000000002
    - type: precision_at_10
      value: 8.233
    - type: precision_at_100
      value: 1.093
    - type: precision_at_1000
      value: 0.11800000000000001
    - type: precision_at_3
      value: 18.579
    - type: precision_at_5
      value: 13.533999999999999
    - type: recall_at_1
      value: 27.602
    - type: recall_at_10
      value: 69.216
    - type: recall_at_100
      value: 90.252
    - type: recall_at_1000
      value: 97.27
    - type: recall_at_3
      value: 47.987
    - type: recall_at_5
      value: 57.438
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
      value: 70.949
    - type: map_at_10
      value: 84.89999999999999
    - type: map_at_100
      value: 85.531
    - type: map_at_1000
      value: 85.548
    - type: map_at_3
      value: 82.027
    - type: map_at_5
      value: 83.853
    - type: mrr_at_1
      value: 81.69999999999999
    - type: mrr_at_10
      value: 87.813
    - type: mrr_at_100
      value: 87.917
    - type: mrr_at_1000
      value: 87.91799999999999
    - type: mrr_at_3
      value: 86.938
    - type: mrr_at_5
      value: 87.53999999999999
    - type: ndcg_at_1
      value: 81.75
    - type: ndcg_at_10
      value: 88.55499999999999
    - type: ndcg_at_100
      value: 89.765
    - type: ndcg_at_1000
      value: 89.871
    - type: ndcg_at_3
      value: 85.905
    - type: ndcg_at_5
      value: 87.41
    - type: precision_at_1
      value: 81.75
    - type: precision_at_10
      value: 13.403
    - type: precision_at_100
      value: 1.528
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.597
    - type: precision_at_5
      value: 24.69
    - type: recall_at_1
      value: 70.949
    - type: recall_at_10
      value: 95.423
    - type: recall_at_100
      value: 99.509
    - type: recall_at_1000
      value: 99.982
    - type: recall_at_3
      value: 87.717
    - type: recall_at_5
      value: 92.032
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
      value: 51.76962893449579
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
      value: 62.32897690686379
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
      value: 4.478
    - type: map_at_10
      value: 11.994
    - type: map_at_100
      value: 13.977
    - type: map_at_1000
      value: 14.295
    - type: map_at_3
      value: 8.408999999999999
    - type: map_at_5
      value: 10.024
    - type: mrr_at_1
      value: 22.1
    - type: mrr_at_10
      value: 33.526
    - type: mrr_at_100
      value: 34.577000000000005
    - type: mrr_at_1000
      value: 34.632000000000005
    - type: mrr_at_3
      value: 30.217
    - type: mrr_at_5
      value: 31.962000000000003
    - type: ndcg_at_1
      value: 22.1
    - type: ndcg_at_10
      value: 20.191
    - type: ndcg_at_100
      value: 27.954
    - type: ndcg_at_1000
      value: 33.491
    - type: ndcg_at_3
      value: 18.787000000000003
    - type: ndcg_at_5
      value: 16.378999999999998
    - type: precision_at_1
      value: 22.1
    - type: precision_at_10
      value: 10.69
    - type: precision_at_100
      value: 2.1919999999999997
    - type: precision_at_1000
      value: 0.35200000000000004
    - type: precision_at_3
      value: 17.732999999999997
    - type: precision_at_5
      value: 14.499999999999998
    - type: recall_at_1
      value: 4.478
    - type: recall_at_10
      value: 21.657
    - type: recall_at_100
      value: 44.54
    - type: recall_at_1000
      value: 71.542
    - type: recall_at_3
      value: 10.778
    - type: recall_at_5
      value: 14.687
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
      value: 82.82325259156718
    - type: cos_sim_spearman
      value: 79.2463589100662
    - type: euclidean_pearson
      value: 80.48318380496771
    - type: euclidean_spearman
      value: 79.34451935199979
    - type: manhattan_pearson
      value: 80.39041824178759
    - type: manhattan_spearman
      value: 79.23002892700211
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
      value: 85.74130231431258
    - type: cos_sim_spearman
      value: 78.36856568042397
    - type: euclidean_pearson
      value: 82.48301631890303
    - type: euclidean_spearman
      value: 78.28376980722732
    - type: manhattan_pearson
      value: 82.43552075450525
    - type: manhattan_spearman
      value: 78.22702443947126
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
      value: 79.96138619461459
    - type: cos_sim_spearman
      value: 81.85436343502379
    - type: euclidean_pearson
      value: 81.82895226665367
    - type: euclidean_spearman
      value: 82.22707349602916
    - type: manhattan_pearson
      value: 81.66303369445873
    - type: manhattan_spearman
      value: 82.05030197179455
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
      value: 80.05481244198648
    - type: cos_sim_spearman
      value: 80.85052504637808
    - type: euclidean_pearson
      value: 80.86728419744497
    - type: euclidean_spearman
      value: 81.033786401512
    - type: manhattan_pearson
      value: 80.90107531061103
    - type: manhattan_spearman
      value: 81.11374116827795
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
      value: 84.615220756399
    - type: cos_sim_spearman
      value: 86.46858500002092
    - type: euclidean_pearson
      value: 86.08307800247586
    - type: euclidean_spearman
      value: 86.72691443870013
    - type: manhattan_pearson
      value: 85.96155594487269
    - type: manhattan_spearman
      value: 86.605909505275
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
      value: 82.14363913634436
    - type: cos_sim_spearman
      value: 84.48430226487102
    - type: euclidean_pearson
      value: 83.75303424801902
    - type: euclidean_spearman
      value: 84.56762380734538
    - type: manhattan_pearson
      value: 83.6135447165928
    - type: manhattan_spearman
      value: 84.39898212616731
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
      value: 85.09909252554525
    - type: cos_sim_spearman
      value: 85.70951402743276
    - type: euclidean_pearson
      value: 87.1991936239908
    - type: euclidean_spearman
      value: 86.07745840612071
    - type: manhattan_pearson
      value: 87.25039137549952
    - type: manhattan_spearman
      value: 85.99938746659761
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
      value: 63.529332093413615
    - type: cos_sim_spearman
      value: 65.38177340147439
    - type: euclidean_pearson
      value: 66.35278011412136
    - type: euclidean_spearman
      value: 65.47147267032997
    - type: manhattan_pearson
      value: 66.71804682408693
    - type: manhattan_spearman
      value: 65.67406521423597
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
      value: 82.45802942885662
    - type: cos_sim_spearman
      value: 84.8853341842566
    - type: euclidean_pearson
      value: 84.60915021096707
    - type: euclidean_spearman
      value: 85.11181242913666
    - type: manhattan_pearson
      value: 84.38600521210364
    - type: manhattan_spearman
      value: 84.89045417981723
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
      value: 85.92793380635129
    - type: mrr
      value: 95.85834191226348
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
      value: 55.74400000000001
    - type: map_at_10
      value: 65.455
    - type: map_at_100
      value: 66.106
    - type: map_at_1000
      value: 66.129
    - type: map_at_3
      value: 62.719
    - type: map_at_5
      value: 64.441
    - type: mrr_at_1
      value: 58.667
    - type: mrr_at_10
      value: 66.776
    - type: mrr_at_100
      value: 67.363
    - type: mrr_at_1000
      value: 67.384
    - type: mrr_at_3
      value: 64.889
    - type: mrr_at_5
      value: 66.122
    - type: ndcg_at_1
      value: 58.667
    - type: ndcg_at_10
      value: 69.904
    - type: ndcg_at_100
      value: 72.807
    - type: ndcg_at_1000
      value: 73.423
    - type: ndcg_at_3
      value: 65.405
    - type: ndcg_at_5
      value: 67.86999999999999
    - type: precision_at_1
      value: 58.667
    - type: precision_at_10
      value: 9.3
    - type: precision_at_100
      value: 1.08
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 25.444
    - type: precision_at_5
      value: 17
    - type: recall_at_1
      value: 55.74400000000001
    - type: recall_at_10
      value: 82.122
    - type: recall_at_100
      value: 95.167
    - type: recall_at_1000
      value: 100
    - type: recall_at_3
      value: 70.14399999999999
    - type: recall_at_5
      value: 76.417
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
      value: 99.86534653465347
    - type: cos_sim_ap
      value: 96.54142419791388
    - type: cos_sim_f1
      value: 93.07535641547861
    - type: cos_sim_precision
      value: 94.81327800829875
    - type: cos_sim_recall
      value: 91.4
    - type: dot_accuracy
      value: 99.86435643564356
    - type: dot_ap
      value: 96.53682260449868
    - type: dot_f1
      value: 92.98515104966718
    - type: dot_precision
      value: 95.27806925498426
    - type: dot_recall
      value: 90.8
    - type: euclidean_accuracy
      value: 99.86336633663366
    - type: euclidean_ap
      value: 96.5228676185697
    - type: euclidean_f1
      value: 92.9735234215886
    - type: euclidean_precision
      value: 94.70954356846472
    - type: euclidean_recall
      value: 91.3
    - type: manhattan_accuracy
      value: 99.85841584158416
    - type: manhattan_ap
      value: 96.50392760934032
    - type: manhattan_f1
      value: 92.84642321160581
    - type: manhattan_precision
      value: 92.8928928928929
    - type: manhattan_recall
      value: 92.80000000000001
    - type: max_accuracy
      value: 99.86534653465347
    - type: max_ap
      value: 96.54142419791388
    - type: max_f1
      value: 93.07535641547861
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
      value: 61.08285408766616
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
      value: 35.640675309010604
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
      value: 53.20333913710715
    - type: mrr
      value: 54.088813555725324
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
      value: 30.79465221925075
    - type: cos_sim_spearman
      value: 30.530816059163634
    - type: dot_pearson
      value: 31.364837244718043
    - type: dot_spearman
      value: 30.79726823684003
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
      value: 0.22599999999999998
    - type: map_at_10
      value: 1.735
    - type: map_at_100
      value: 8.978
    - type: map_at_1000
      value: 20.851
    - type: map_at_3
      value: 0.613
    - type: map_at_5
      value: 0.964
    - type: mrr_at_1
      value: 88
    - type: mrr_at_10
      value: 92.867
    - type: mrr_at_100
      value: 92.867
    - type: mrr_at_1000
      value: 92.867
    - type: mrr_at_3
      value: 92.667
    - type: mrr_at_5
      value: 92.667
    - type: ndcg_at_1
      value: 82
    - type: ndcg_at_10
      value: 73.164
    - type: ndcg_at_100
      value: 51.878
    - type: ndcg_at_1000
      value: 44.864
    - type: ndcg_at_3
      value: 79.184
    - type: ndcg_at_5
      value: 76.39
    - type: precision_at_1
      value: 88
    - type: precision_at_10
      value: 76.2
    - type: precision_at_100
      value: 52.459999999999994
    - type: precision_at_1000
      value: 19.692
    - type: precision_at_3
      value: 82.667
    - type: precision_at_5
      value: 80
    - type: recall_at_1
      value: 0.22599999999999998
    - type: recall_at_10
      value: 1.942
    - type: recall_at_100
      value: 12.342
    - type: recall_at_1000
      value: 41.42
    - type: recall_at_3
      value: 0.637
    - type: recall_at_5
      value: 1.034
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
      value: 3.567
    - type: map_at_10
      value: 13.116
    - type: map_at_100
      value: 19.39
    - type: map_at_1000
      value: 20.988
    - type: map_at_3
      value: 7.109
    - type: map_at_5
      value: 9.950000000000001
    - type: mrr_at_1
      value: 42.857
    - type: mrr_at_10
      value: 57.404999999999994
    - type: mrr_at_100
      value: 58.021
    - type: mrr_at_1000
      value: 58.021
    - type: mrr_at_3
      value: 54.762
    - type: mrr_at_5
      value: 56.19
    - type: ndcg_at_1
      value: 38.775999999999996
    - type: ndcg_at_10
      value: 30.359
    - type: ndcg_at_100
      value: 41.284
    - type: ndcg_at_1000
      value: 52.30200000000001
    - type: ndcg_at_3
      value: 36.744
    - type: ndcg_at_5
      value: 34.326
    - type: precision_at_1
      value: 42.857
    - type: precision_at_10
      value: 26.122
    - type: precision_at_100
      value: 8.082
    - type: precision_at_1000
      value: 1.559
    - type: precision_at_3
      value: 40.136
    - type: precision_at_5
      value: 35.510000000000005
    - type: recall_at_1
      value: 3.567
    - type: recall_at_10
      value: 19.045
    - type: recall_at_100
      value: 49.979
    - type: recall_at_1000
      value: 84.206
    - type: recall_at_3
      value: 8.52
    - type: recall_at_5
      value: 13.103000000000002
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
      value: 68.8394
    - type: ap
      value: 13.454399712443099
    - type: f1
      value: 53.04963076364322
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
      value: 60.546123372948514
    - type: f1
      value: 60.86952793277713
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
      value: 49.10042955060234
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
      value: 85.03308100375514
    - type: cos_sim_ap
      value: 71.08284605869684
    - type: cos_sim_f1
      value: 65.42539436255494
    - type: cos_sim_precision
      value: 64.14807302231237
    - type: cos_sim_recall
      value: 66.75461741424802
    - type: dot_accuracy
      value: 84.68736961316088
    - type: dot_ap
      value: 69.20524036530992
    - type: dot_f1
      value: 63.54893953365829
    - type: dot_precision
      value: 63.45698500394633
    - type: dot_recall
      value: 63.641160949868066
    - type: euclidean_accuracy
      value: 85.07480479227513
    - type: euclidean_ap
      value: 71.14592761009864
    - type: euclidean_f1
      value: 65.43814432989691
    - type: euclidean_precision
      value: 63.95465994962216
    - type: euclidean_recall
      value: 66.99208443271768
    - type: manhattan_accuracy
      value: 85.06288370984085
    - type: manhattan_ap
      value: 71.07289742593868
    - type: manhattan_f1
      value: 65.37585421412301
    - type: manhattan_precision
      value: 62.816147859922175
    - type: manhattan_recall
      value: 68.15303430079156
    - type: max_accuracy
      value: 85.07480479227513
    - type: max_ap
      value: 71.14592761009864
    - type: max_f1
      value: 65.43814432989691
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
      value: 87.79058485659952
    - type: cos_sim_ap
      value: 83.7183187008759
    - type: cos_sim_f1
      value: 75.86921142180798
    - type: cos_sim_precision
      value: 73.00683371298405
    - type: cos_sim_recall
      value: 78.96519864490298
    - type: dot_accuracy
      value: 87.0085768618776
    - type: dot_ap
      value: 81.87467488474279
    - type: dot_f1
      value: 74.04188363990559
    - type: dot_precision
      value: 72.10507114191901
    - type: dot_recall
      value: 76.08561749307053
    - type: euclidean_accuracy
      value: 87.8332751193387
    - type: euclidean_ap
      value: 83.83585648120315
    - type: euclidean_f1
      value: 76.02582177042369
    - type: euclidean_precision
      value: 73.36388371759989
    - type: euclidean_recall
      value: 78.88820449645827
    - type: manhattan_accuracy
      value: 87.87208444910156
    - type: manhattan_ap
      value: 83.8101950642973
    - type: manhattan_f1
      value: 75.90454195535027
    - type: manhattan_precision
      value: 72.44419564761039
    - type: manhattan_recall
      value: 79.71204188481676
    - type: max_accuracy
      value: 87.87208444910156
    - type: max_ap
      value: 83.83585648120315
    - type: max_f1
      value: 76.02582177042369
---


**Recommend switching to newest [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), which has more reasonable similarity distribution and same method of usage.**

<h1 align="center">FlagEmbedding</h1>


<h4 align="center">
    <p>
        <a href=#model-list>Model List</a> | 
        <a href=#frequently-asked-questions>FAQ</a> |
        <a href=#usage>Usage</a>  |
        <a href="#evaluation">Evaluation</a> |
        <a href="#train">Train</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

More details please refer to our Github: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).


[English](README.md) | [中文](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

FlagEmbedding focus on retrieval-augmented LLMs, consisting of following projects currently:

- **Fine-tuning of LM** : [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)
- **Dense Retrieval**: [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding), [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- **Reranker Model**: [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)


## News 

- 11/23/2023: Release [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail), a method to maintain general capabilities during fine-tuning by merging multiple language models. [Technical Report](https://arxiv.org/abs/2311.13534) :fire:  
- 10/12/2023: Release [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), a unified embedding model to support diverse retrieval augmentation needs for LLMs. [Technical Report](https://arxiv.org/pdf/2310.07554.pdf)
- 09/15/2023: The [technical report](https://arxiv.org/pdf/2309.07597.pdf) of BGE has been released 
- 09/15/2023: The [massive training data](https://data.baai.ac.cn/details/BAAI-MTP) of BGE has been released 
- 09/12/2023: New models: 
    - **New reranker model**: release cross-encoder models `BAAI/bge-reranker-base` and `BAAI/bge-reranker-large`, which are more powerful than embedding model. We recommend to use/fine-tune them to re-rank top-k documents returned by embedding models. 
    - **update embedding model**: release `bge-*-v1.5` embedding model to alleviate the issue of the similarity distribution, and enhance its retrieval ability without instruction.
 

<details>
  <summary>More</summary>
<!-- ### More -->
    
- 09/07/2023: Update [fine-tune code](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md): Add script to mine hard negatives and support adding instruction during fine-tuning. 
- 08/09/2023: BGE Models are integrated into **Langchain**, you can use it like [this](#using-langchain); C-MTEB **leaderboard** is [available](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: Release base-scale and small-scale models, **best performance among the models of the same size 🤗**  
- 08/02/2023: Release `bge-large-*`(short for BAAI General Embedding) Models, **rank 1st on MTEB and C-MTEB benchmark!** :tada: :tada:   
- 08/01/2023: We release the [Chinese Massive Text Embedding Benchmark](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), consisting of 31 test dataset.  
  
</details>


## Model List

`bge` is short for `BAAI general embedding`.

|              Model              | Language | | Description | query instruction for retrieval [1] |
|:-------------------------------|:--------:| :--------:| :--------:|:--------:|
| [LM-Cocktail](https://huggingface.co/Shitao)                   |   English |  | fine-tuned models (Llama and BGE) which can be used to reproduce the results of LM-Cocktail |  |
|  [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)  |   English | [Inference](./FlagEmbedding/llm_embedder/README.md) [Fine-tune](./FlagEmbedding/llm_embedder/README.md) | a unified embedding model to support diverse retrieval augmentation needs for LLMs | See [README](./FlagEmbedding/llm_embedder/README.md) |
|  [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)  |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | a cross-encoder model which is more accurate but less efficient [2] |   |
|  [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | a cross-encoder model which is more accurate but less efficient [2] |   |
|  [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a base-scale model but with similar ability to `bge-large-en` | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |a small-scale model but with competitive performance  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a base-scale model but with similar ability to `bge-large-zh` | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a small-scale model but with competitive performance | `为这个句子生成表示以用于检索相关文章：`  |


[1\]: If you need to search the relevant passages to a query, we suggest to add the instruction to the query; in other cases, no instruction is needed, just use the original query directly. In all cases, **no instruction** needs to be added to passages.

[2\]: Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. To balance the accuracy and time cost, cross-encoder is widely used to re-rank top-k documents retrieved by other simple models. 
For examples, use bge embedding model to retrieve top 100 relevant documents, and then use bge reranker to re-rank the top 100 document to get the final top-3 results.

All models have been uploaded to Huggingface Hub, and you can see them at https://huggingface.co/BAAI. 
If you cannot open the Huggingface Hub, you also can download the models at https://model.baai.ac.cn/models .


## Frequently asked questions

<details>
  <summary>1. How to fine-tune bge embedding model?</summary>

  <!-- ### How to fine-tune bge embedding model? -->
Following this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) to prepare data and fine-tune your model. 
Some suggestions:
- Mine hard negatives following this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives), which can improve the retrieval performance.
- If you pre-train bge on your data, the pre-trained model cannot be directly used to calculate similarity, and it must be fine-tuned with contrastive learning before computing similarity.
- If the accuracy of the fine-tuned model is still not high, it is recommended to use/fine-tune the cross-encoder model (bge-reranker) to re-rank top-k results. Hard negatives also are needed to fine-tune reranker.

  
</details>

<details>
  <summary>2. The similarity score between two dissimilar sentences is higher than 0.5</summary>

  <!-- ### The similarity score between two dissimilar sentences is higher than 0.5 -->
**Suggest to use bge v1.5, which alleviates the issue of the similarity distribution.** 

Since we finetune the models by contrastive learning with a temperature of 0.01, 
the similarity distribution of the current BGE model is about in the interval \[0.6, 1\].
So a similarity score greater than 0.5 does not indicate that the two sentences are similar.

For downstream tasks, such as passage retrieval or semantic similarity, 
**what matters is the relative order of the scores, not the absolute value.**
If you need to filter similar sentences based on a similarity threshold, 
please select an appropriate similarity threshold based on the similarity distribution on your data (such as 0.8, 0.85, or even 0.9).

</details>

<details>
  <summary>3. When does the query instruction need to be used</summary>

  <!-- ### When does the query instruction need to be used -->

For the `bge-*-v1.5`, we improve its retrieval ability when not using instruction. 
No instruction only has a slight degradation in retrieval performance compared with using instruction. 
So you can generate embedding without instruction in all cases for convenience.
 
For a retrieval task that uses short queries to find long related documents, 
it is recommended to add instructions for these short queries.
**The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task.**
In all cases, the documents/passages do not need to add the instruction. 

</details>


## Usage 

### Usage for Embedding Model

Here are some examples for using `bge` models with 
[FlagEmbedding](#using-flagembedding), [Sentence-Transformers](#using-sentence-transformers), [Langchain](#using-langchain), or [Huggingface Transformers](#using-huggingface-transformers).

#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```
If it doesn't work for you, you can see [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) for more methods to install FlagEmbedding.

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```
For the value of the argument `query_instruction_for_retrieval`, see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list). 

By default, FlagModel will use all available GPUs when encoding. Please set `os.environ["CUDA_VISIBLE_DEVICES"]` to select specific GPUs.
You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make all GPUs unavailable.


#### Using Sentence-Transformers

You can also use the `bge` models with [sentence-transformers](https://www.SBERT.net):

```
pip install -U sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```
For s2p(short query to long passage) retrieval task, 
each short query should start with an instruction (instructions see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). 
But the instruction is not needed for passages.
```python
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

#### Using Langchain 

You can use `bge` in langchain like this:
```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："
```


#### Using HuggingFace Transformers

With the transformers package, you can use the model like this: First, you pass your input through the transformer model, then you select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```

### Usage for Reranker

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. 
You can get a relevance score by inputting query and passage to the reranker. 
The reranker is optimized based cross-entropy loss, so the relevance score is not bounded to a specific range.


#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```

Get relevance scores (higher scores indicate more relevance):
```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```


#### Using Huggingface transformers

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```

## Evaluation  

`baai-general-embedding` models achieve **state-of-the-art performance on both MTEB and C-MTEB leaderboard!**
For more details and evaluation tools see our [scripts](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md). 

- **MTEB**:   

| Model Name |  Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1024 | 512 |  **64.23** | **54.29** |  46.08 | 87.12 | 60.03 | 83.11 | 31.61 | 75.97 |  
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |  768 | 512 | 63.55 | 53.25 |   45.77 | 86.55 | 58.86 | 82.4 | 31.07 | 75.53 |  
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |  384 | 512 | 62.17 |51.68 | 43.82 |  84.92 | 58.36 | 81.59 | 30.12 | 74.14 |  
| [bge-large-en](https://huggingface.co/BAAI/bge-large-en) |  1024 | 512 | 63.98 |  53.9 | 46.98 | 85.8 | 59.48 | 81.56 | 32.06 | 76.21 | 
| [bge-base-en](https://huggingface.co/BAAI/bge-base-en) |  768 | 512 |  63.36 | 53.0 | 46.32 | 85.86 | 58.7 | 81.84 | 29.27 | 75.27 | 
| [gte-large](https://huggingface.co/thenlper/gte-large) |  1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base) 	|  768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) |  1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [bge-small-en](https://huggingface.co/BAAI/bge-small-en) |  384 | 512 | 62.11 |  51.82 | 44.31 | 83.78 | 57.97 | 80.72 | 30.53 | 74.37 |  
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) |  768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) |  768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small) |  384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) |  768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	|  768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	|  4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |



- **C-MTEB**:  
We create the benchmark C-MTEB for Chinese text embedding which consists of 31 datasets from 6 tasks. 
Please refer to [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md) for a detailed introduction.
 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 1024 |  **64.53** | 70.46 | 56.25 | 81.6 | 69.13 | 65.84 | 48.99 |  
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 768 |  63.13 | 69.49 | 53.72 | 79.75 | 68.07 | 65.39 | 47.53 |  
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 512 | 57.82 | 61.77 | 49.11 | 70.41 | 63.96 | 60.92 | 44.18 |   
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | 1024 | 64.20 | 71.53 | 54.98 | 78.94 | 68.32 | 65.11 | 48.39 |
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 53 | 76.77 | 68.58 | 64.91 | 50.01 |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 768 | 62.96 | 69.53 | 54.12 | 77.5 | 67.07 | 64.91 | 47.63 |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | 58.79 | 63.66 | 48.44 | 69.89 | 67.34 | 56.00 | 48.23 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 512 | 58.27 |  63.07 | 49.45 | 70.35 | 63.64 | 61.48 | 45.09 |
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 | 56.91 | 50.47 | 63.99 | 67.52 | 59.34 | 47.68 |
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 | 54.75 | 50.42 | 64.3 | 68.2 | 59.66 | 48.88 |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 768 | 55.48 | 61.63 | 46.49 | 67.07 | 65.35 | 54.35 | 40.68 |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 384 | 55.38 | 59.95 | 45.27 | 66.45 | 65.85 | 53.86 | 45.26 |
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 43.35 | 69.56 | 64.31 | 54.28 | 45.68 |
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 42.78 | 66.62 | 61 | 49.25 | 44.39 |
| [text2vec-base](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 43.41 | 67.41 | 62.19 | 49.45 | 37.66 |
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 44.97 | 70.86 | 60.66 | 49.16 | 30.02 |


- **Reranking**:
See [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/) for evaluation script.

| Model | T2Reranking | T2RerankingZh2En\* | T2RerankingEn2Zh\* | MMarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| text2vec-base-multilingual | 64.66 | 62.94 | 62.51 | 14.37 | 48.46 | 48.6 | 50.26 |  
| multilingual-e5-small | 65.62 | 60.94 | 56.41 | 29.91 | 67.26 | 66.54 | 57.78 |  
| multilingual-e5-large | 64.55 | 61.61 | 54.28 | 28.6 | 67.42 | 67.92 | 57.4 |  
| multilingual-e5-base | 64.21 | 62.13 | 54.68 | 29.5 | 66.23 | 66.98 | 57.29 |  
| m3e-base | 66.03 | 62.74 | 56.07 | 17.51 | 77.05 | 76.76 | 59.36 |  
| m3e-large | 66.13 | 62.72 | 56.1 | 16.46 | 77.76 | 78.27 | 59.57 |  
| bge-base-zh-v1.5 | 66.49 | 63.25 | 57.02 | 29.74 | 80.47 | 84.88 | 63.64 |  
| bge-large-zh-v1.5 | 65.74 | 63.39 | 57.03 | 28.74 | 83.45 | 85.44 | 63.97 |  
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | 67.28 | 63.95 | 60.45 | 35.46 | 81.26 | 84.1 | 65.42 |  
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 67.6 | 64.03 | 61.44 | 37.16 | 82.15 | 84.18 | 66.09 |  

\* : T2RerankingZh2En and T2RerankingEn2Zh are cross-language retrieval tasks

## Train

### BAAI Embedding 

We pre-train the models using [retromae](https://github.com/staoxiao/RetroMAE) and train them on large-scale pairs data using contrastive learning. 
**You can fine-tune the embedding model on your data following our [examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune).**
We also provide a [pre-train example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain).
Note that the goal of pre-training is to reconstruct the text, and the pre-trained model cannot be used for similarity calculation directly, it needs to be fine-tuned.
More training details for bge see [baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md).



### BGE Reranker

Cross-encoder will perform full-attention over the input pair, 
which is more accurate than embedding model (i.e., bi-encoder) but more time-consuming than embedding model.
Therefore, it can be used to re-rank the top-k documents returned by embedding model.
We train the cross-encoder on a multilingual pair data, 
The data format is the same as embedding model, so you can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker). 
More details please refer to [./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)




## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlagEmbedding is licensed under the [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). The released models can be used for commercial purposes free of charge.

