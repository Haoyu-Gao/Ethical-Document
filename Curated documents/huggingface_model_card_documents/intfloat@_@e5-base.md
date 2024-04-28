---
language:
- en
license: mit
tags:
- mteb
- Sentence Transformers
- sentence-similarity
- sentence-transformers
model-index:
- name: e5-base
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
      value: 79.71641791044777
    - type: ap
      value: 44.15426065428253
    - type: f1
      value: 73.89474407693241
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
      value: 87.9649
    - type: ap
      value: 84.10171551915973
    - type: f1
      value: 87.94148377827356
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
      value: 42.645999999999994
    - type: f1
      value: 42.230574673549
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
      value: 26.814
    - type: map_at_10
      value: 42.681999999999995
    - type: map_at_100
      value: 43.714
    - type: map_at_1000
      value: 43.724000000000004
    - type: map_at_3
      value: 38.11
    - type: map_at_5
      value: 40.666999999999994
    - type: mrr_at_1
      value: 27.168999999999997
    - type: mrr_at_10
      value: 42.84
    - type: mrr_at_100
      value: 43.864
    - type: mrr_at_1000
      value: 43.875
    - type: mrr_at_3
      value: 38.193
    - type: mrr_at_5
      value: 40.793
    - type: ndcg_at_1
      value: 26.814
    - type: ndcg_at_10
      value: 51.410999999999994
    - type: ndcg_at_100
      value: 55.713
    - type: ndcg_at_1000
      value: 55.957
    - type: ndcg_at_3
      value: 41.955
    - type: ndcg_at_5
      value: 46.558
    - type: precision_at_1
      value: 26.814
    - type: precision_at_10
      value: 7.922999999999999
    - type: precision_at_100
      value: 0.9780000000000001
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 17.71
    - type: precision_at_5
      value: 12.859000000000002
    - type: recall_at_1
      value: 26.814
    - type: recall_at_10
      value: 79.232
    - type: recall_at_100
      value: 97.795
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 53.129000000000005
    - type: recall_at_5
      value: 64.29599999999999
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
      value: 44.56933066536439
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
      value: 40.47647746165173
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
      value: 59.65675531567043
    - type: mrr
      value: 72.95255683067317
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
      value: 85.83147014162338
    - type: cos_sim_spearman
      value: 85.1031439521441
    - type: euclidean_pearson
      value: 83.53609085510973
    - type: euclidean_spearman
      value: 84.59650590202833
    - type: manhattan_pearson
      value: 83.14611947586386
    - type: manhattan_spearman
      value: 84.13384475757064
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
      value: 83.32792207792208
    - type: f1
      value: 83.32037485050513
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
      value: 36.18605446588703
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
      value: 32.72379130181917
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
      value: 30.659
    - type: map_at_10
      value: 40.333999999999996
    - type: map_at_100
      value: 41.763
    - type: map_at_1000
      value: 41.894
    - type: map_at_3
      value: 37.561
    - type: map_at_5
      value: 39.084
    - type: mrr_at_1
      value: 37.482
    - type: mrr_at_10
      value: 45.736
    - type: mrr_at_100
      value: 46.591
    - type: mrr_at_1000
      value: 46.644999999999996
    - type: mrr_at_3
      value: 43.491
    - type: mrr_at_5
      value: 44.75
    - type: ndcg_at_1
      value: 37.482
    - type: ndcg_at_10
      value: 45.606
    - type: ndcg_at_100
      value: 51.172
    - type: ndcg_at_1000
      value: 53.407000000000004
    - type: ndcg_at_3
      value: 41.808
    - type: ndcg_at_5
      value: 43.449
    - type: precision_at_1
      value: 37.482
    - type: precision_at_10
      value: 8.254999999999999
    - type: precision_at_100
      value: 1.3719999999999999
    - type: precision_at_1000
      value: 0.186
    - type: precision_at_3
      value: 19.695
    - type: precision_at_5
      value: 13.847999999999999
    - type: recall_at_1
      value: 30.659
    - type: recall_at_10
      value: 55.409
    - type: recall_at_100
      value: 78.687
    - type: recall_at_1000
      value: 93.068
    - type: recall_at_3
      value: 43.891999999999996
    - type: recall_at_5
      value: 48.678
    - type: map_at_1
      value: 30.977
    - type: map_at_10
      value: 40.296
    - type: map_at_100
      value: 41.453
    - type: map_at_1000
      value: 41.581
    - type: map_at_3
      value: 37.619
    - type: map_at_5
      value: 39.181
    - type: mrr_at_1
      value: 39.108
    - type: mrr_at_10
      value: 46.894000000000005
    - type: mrr_at_100
      value: 47.55
    - type: mrr_at_1000
      value: 47.598
    - type: mrr_at_3
      value: 44.766
    - type: mrr_at_5
      value: 46.062999999999995
    - type: ndcg_at_1
      value: 39.108
    - type: ndcg_at_10
      value: 45.717
    - type: ndcg_at_100
      value: 49.941
    - type: ndcg_at_1000
      value: 52.138
    - type: ndcg_at_3
      value: 42.05
    - type: ndcg_at_5
      value: 43.893
    - type: precision_at_1
      value: 39.108
    - type: precision_at_10
      value: 8.306
    - type: precision_at_100
      value: 1.3419999999999999
    - type: precision_at_1000
      value: 0.184
    - type: precision_at_3
      value: 19.979
    - type: precision_at_5
      value: 14.038
    - type: recall_at_1
      value: 30.977
    - type: recall_at_10
      value: 54.688
    - type: recall_at_100
      value: 72.556
    - type: recall_at_1000
      value: 86.53800000000001
    - type: recall_at_3
      value: 43.388
    - type: recall_at_5
      value: 48.717
    - type: map_at_1
      value: 39.812
    - type: map_at_10
      value: 50.1
    - type: map_at_100
      value: 51.193999999999996
    - type: map_at_1000
      value: 51.258
    - type: map_at_3
      value: 47.510999999999996
    - type: map_at_5
      value: 48.891
    - type: mrr_at_1
      value: 45.266
    - type: mrr_at_10
      value: 53.459999999999994
    - type: mrr_at_100
      value: 54.19199999999999
    - type: mrr_at_1000
      value: 54.228
    - type: mrr_at_3
      value: 51.296
    - type: mrr_at_5
      value: 52.495999999999995
    - type: ndcg_at_1
      value: 45.266
    - type: ndcg_at_10
      value: 55.034000000000006
    - type: ndcg_at_100
      value: 59.458
    - type: ndcg_at_1000
      value: 60.862
    - type: ndcg_at_3
      value: 50.52799999999999
    - type: ndcg_at_5
      value: 52.564
    - type: precision_at_1
      value: 45.266
    - type: precision_at_10
      value: 8.483
    - type: precision_at_100
      value: 1.162
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 21.944
    - type: precision_at_5
      value: 14.721
    - type: recall_at_1
      value: 39.812
    - type: recall_at_10
      value: 66.36
    - type: recall_at_100
      value: 85.392
    - type: recall_at_1000
      value: 95.523
    - type: recall_at_3
      value: 54.127
    - type: recall_at_5
      value: 59.245000000000005
    - type: map_at_1
      value: 26.186
    - type: map_at_10
      value: 33.18
    - type: map_at_100
      value: 34.052
    - type: map_at_1000
      value: 34.149
    - type: map_at_3
      value: 31.029
    - type: map_at_5
      value: 32.321
    - type: mrr_at_1
      value: 28.136
    - type: mrr_at_10
      value: 35.195
    - type: mrr_at_100
      value: 35.996
    - type: mrr_at_1000
      value: 36.076
    - type: mrr_at_3
      value: 33.051
    - type: mrr_at_5
      value: 34.407
    - type: ndcg_at_1
      value: 28.136
    - type: ndcg_at_10
      value: 37.275999999999996
    - type: ndcg_at_100
      value: 41.935
    - type: ndcg_at_1000
      value: 44.389
    - type: ndcg_at_3
      value: 33.059
    - type: ndcg_at_5
      value: 35.313
    - type: precision_at_1
      value: 28.136
    - type: precision_at_10
      value: 5.457999999999999
    - type: precision_at_100
      value: 0.826
    - type: precision_at_1000
      value: 0.107
    - type: precision_at_3
      value: 13.522
    - type: precision_at_5
      value: 9.424000000000001
    - type: recall_at_1
      value: 26.186
    - type: recall_at_10
      value: 47.961999999999996
    - type: recall_at_100
      value: 70.072
    - type: recall_at_1000
      value: 88.505
    - type: recall_at_3
      value: 36.752
    - type: recall_at_5
      value: 42.168
    - type: map_at_1
      value: 16.586000000000002
    - type: map_at_10
      value: 23.637
    - type: map_at_100
      value: 24.82
    - type: map_at_1000
      value: 24.95
    - type: map_at_3
      value: 21.428
    - type: map_at_5
      value: 22.555
    - type: mrr_at_1
      value: 20.771
    - type: mrr_at_10
      value: 27.839999999999996
    - type: mrr_at_100
      value: 28.887
    - type: mrr_at_1000
      value: 28.967
    - type: mrr_at_3
      value: 25.56
    - type: mrr_at_5
      value: 26.723000000000003
    - type: ndcg_at_1
      value: 20.771
    - type: ndcg_at_10
      value: 28.255000000000003
    - type: ndcg_at_100
      value: 33.886
    - type: ndcg_at_1000
      value: 36.963
    - type: ndcg_at_3
      value: 24.056
    - type: ndcg_at_5
      value: 25.818
    - type: precision_at_1
      value: 20.771
    - type: precision_at_10
      value: 5.1
    - type: precision_at_100
      value: 0.9119999999999999
    - type: precision_at_1000
      value: 0.132
    - type: precision_at_3
      value: 11.526
    - type: precision_at_5
      value: 8.158999999999999
    - type: recall_at_1
      value: 16.586000000000002
    - type: recall_at_10
      value: 38.456
    - type: recall_at_100
      value: 62.666
    - type: recall_at_1000
      value: 84.47
    - type: recall_at_3
      value: 26.765
    - type: recall_at_5
      value: 31.297000000000004
    - type: map_at_1
      value: 28.831
    - type: map_at_10
      value: 37.545
    - type: map_at_100
      value: 38.934999999999995
    - type: map_at_1000
      value: 39.044000000000004
    - type: map_at_3
      value: 34.601
    - type: map_at_5
      value: 36.302
    - type: mrr_at_1
      value: 34.264
    - type: mrr_at_10
      value: 42.569
    - type: mrr_at_100
      value: 43.514
    - type: mrr_at_1000
      value: 43.561
    - type: mrr_at_3
      value: 40.167
    - type: mrr_at_5
      value: 41.678
    - type: ndcg_at_1
      value: 34.264
    - type: ndcg_at_10
      value: 42.914
    - type: ndcg_at_100
      value: 48.931999999999995
    - type: ndcg_at_1000
      value: 51.004000000000005
    - type: ndcg_at_3
      value: 38.096999999999994
    - type: ndcg_at_5
      value: 40.509
    - type: precision_at_1
      value: 34.264
    - type: precision_at_10
      value: 7.642
    - type: precision_at_100
      value: 1.258
    - type: precision_at_1000
      value: 0.161
    - type: precision_at_3
      value: 17.453
    - type: precision_at_5
      value: 12.608
    - type: recall_at_1
      value: 28.831
    - type: recall_at_10
      value: 53.56999999999999
    - type: recall_at_100
      value: 79.26100000000001
    - type: recall_at_1000
      value: 92.862
    - type: recall_at_3
      value: 40.681
    - type: recall_at_5
      value: 46.597
    - type: map_at_1
      value: 27.461000000000002
    - type: map_at_10
      value: 35.885
    - type: map_at_100
      value: 37.039
    - type: map_at_1000
      value: 37.16
    - type: map_at_3
      value: 33.451
    - type: map_at_5
      value: 34.807
    - type: mrr_at_1
      value: 34.018
    - type: mrr_at_10
      value: 41.32
    - type: mrr_at_100
      value: 42.157
    - type: mrr_at_1000
      value: 42.223
    - type: mrr_at_3
      value: 39.288000000000004
    - type: mrr_at_5
      value: 40.481
    - type: ndcg_at_1
      value: 34.018
    - type: ndcg_at_10
      value: 40.821000000000005
    - type: ndcg_at_100
      value: 46.053
    - type: ndcg_at_1000
      value: 48.673
    - type: ndcg_at_3
      value: 36.839
    - type: ndcg_at_5
      value: 38.683
    - type: precision_at_1
      value: 34.018
    - type: precision_at_10
      value: 7.009
    - type: precision_at_100
      value: 1.123
    - type: precision_at_1000
      value: 0.153
    - type: precision_at_3
      value: 16.933
    - type: precision_at_5
      value: 11.826
    - type: recall_at_1
      value: 27.461000000000002
    - type: recall_at_10
      value: 50.285000000000004
    - type: recall_at_100
      value: 73.25500000000001
    - type: recall_at_1000
      value: 91.17699999999999
    - type: recall_at_3
      value: 39.104
    - type: recall_at_5
      value: 43.968
    - type: map_at_1
      value: 26.980083333333337
    - type: map_at_10
      value: 34.47208333333333
    - type: map_at_100
      value: 35.609249999999996
    - type: map_at_1000
      value: 35.72833333333333
    - type: map_at_3
      value: 32.189416666666666
    - type: map_at_5
      value: 33.44683333333334
    - type: mrr_at_1
      value: 31.731666666666662
    - type: mrr_at_10
      value: 38.518
    - type: mrr_at_100
      value: 39.38166666666667
    - type: mrr_at_1000
      value: 39.446999999999996
    - type: mrr_at_3
      value: 36.49966666666668
    - type: mrr_at_5
      value: 37.639916666666664
    - type: ndcg_at_1
      value: 31.731666666666662
    - type: ndcg_at_10
      value: 38.92033333333333
    - type: ndcg_at_100
      value: 44.01675
    - type: ndcg_at_1000
      value: 46.51075
    - type: ndcg_at_3
      value: 35.09766666666667
    - type: ndcg_at_5
      value: 36.842999999999996
    - type: precision_at_1
      value: 31.731666666666662
    - type: precision_at_10
      value: 6.472583333333332
    - type: precision_at_100
      value: 1.0665
    - type: precision_at_1000
      value: 0.14725000000000002
    - type: precision_at_3
      value: 15.659083333333331
    - type: precision_at_5
      value: 10.878833333333333
    - type: recall_at_1
      value: 26.980083333333337
    - type: recall_at_10
      value: 48.13925
    - type: recall_at_100
      value: 70.70149999999998
    - type: recall_at_1000
      value: 88.10775000000001
    - type: recall_at_3
      value: 37.30091666666667
    - type: recall_at_5
      value: 41.90358333333333
    - type: map_at_1
      value: 25.607999999999997
    - type: map_at_10
      value: 30.523
    - type: map_at_100
      value: 31.409
    - type: map_at_1000
      value: 31.507
    - type: map_at_3
      value: 28.915000000000003
    - type: map_at_5
      value: 29.756
    - type: mrr_at_1
      value: 28.681
    - type: mrr_at_10
      value: 33.409
    - type: mrr_at_100
      value: 34.241
    - type: mrr_at_1000
      value: 34.313
    - type: mrr_at_3
      value: 32.029999999999994
    - type: mrr_at_5
      value: 32.712
    - type: ndcg_at_1
      value: 28.681
    - type: ndcg_at_10
      value: 33.733000000000004
    - type: ndcg_at_100
      value: 38.32
    - type: ndcg_at_1000
      value: 40.937
    - type: ndcg_at_3
      value: 30.898999999999997
    - type: ndcg_at_5
      value: 32.088
    - type: precision_at_1
      value: 28.681
    - type: precision_at_10
      value: 4.968999999999999
    - type: precision_at_100
      value: 0.79
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 12.73
    - type: precision_at_5
      value: 8.558
    - type: recall_at_1
      value: 25.607999999999997
    - type: recall_at_10
      value: 40.722
    - type: recall_at_100
      value: 61.956999999999994
    - type: recall_at_1000
      value: 81.43
    - type: recall_at_3
      value: 32.785
    - type: recall_at_5
      value: 35.855
    - type: map_at_1
      value: 20.399
    - type: map_at_10
      value: 25.968000000000004
    - type: map_at_100
      value: 26.985999999999997
    - type: map_at_1000
      value: 27.105
    - type: map_at_3
      value: 24.215
    - type: map_at_5
      value: 25.157
    - type: mrr_at_1
      value: 24.708
    - type: mrr_at_10
      value: 29.971999999999998
    - type: mrr_at_100
      value: 30.858
    - type: mrr_at_1000
      value: 30.934
    - type: mrr_at_3
      value: 28.304000000000002
    - type: mrr_at_5
      value: 29.183999999999997
    - type: ndcg_at_1
      value: 24.708
    - type: ndcg_at_10
      value: 29.676000000000002
    - type: ndcg_at_100
      value: 34.656
    - type: ndcg_at_1000
      value: 37.588
    - type: ndcg_at_3
      value: 26.613
    - type: ndcg_at_5
      value: 27.919
    - type: precision_at_1
      value: 24.708
    - type: precision_at_10
      value: 5.01
    - type: precision_at_100
      value: 0.876
    - type: precision_at_1000
      value: 0.13
    - type: precision_at_3
      value: 11.975
    - type: precision_at_5
      value: 8.279
    - type: recall_at_1
      value: 20.399
    - type: recall_at_10
      value: 36.935
    - type: recall_at_100
      value: 59.532
    - type: recall_at_1000
      value: 80.58
    - type: recall_at_3
      value: 27.979
    - type: recall_at_5
      value: 31.636999999999997
    - type: map_at_1
      value: 27.606
    - type: map_at_10
      value: 34.213
    - type: map_at_100
      value: 35.339999999999996
    - type: map_at_1000
      value: 35.458
    - type: map_at_3
      value: 31.987
    - type: map_at_5
      value: 33.322
    - type: mrr_at_1
      value: 31.53
    - type: mrr_at_10
      value: 37.911
    - type: mrr_at_100
      value: 38.879000000000005
    - type: mrr_at_1000
      value: 38.956
    - type: mrr_at_3
      value: 35.868
    - type: mrr_at_5
      value: 37.047999999999995
    - type: ndcg_at_1
      value: 31.53
    - type: ndcg_at_10
      value: 38.312000000000005
    - type: ndcg_at_100
      value: 43.812
    - type: ndcg_at_1000
      value: 46.414
    - type: ndcg_at_3
      value: 34.319
    - type: ndcg_at_5
      value: 36.312
    - type: precision_at_1
      value: 31.53
    - type: precision_at_10
      value: 5.970000000000001
    - type: precision_at_100
      value: 0.9939999999999999
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 14.738999999999999
    - type: precision_at_5
      value: 10.242999999999999
    - type: recall_at_1
      value: 27.606
    - type: recall_at_10
      value: 47.136
    - type: recall_at_100
      value: 71.253
    - type: recall_at_1000
      value: 89.39399999999999
    - type: recall_at_3
      value: 36.342
    - type: recall_at_5
      value: 41.388999999999996
    - type: map_at_1
      value: 24.855
    - type: map_at_10
      value: 31.963
    - type: map_at_100
      value: 33.371
    - type: map_at_1000
      value: 33.584
    - type: map_at_3
      value: 29.543999999999997
    - type: map_at_5
      value: 30.793
    - type: mrr_at_1
      value: 29.644
    - type: mrr_at_10
      value: 35.601
    - type: mrr_at_100
      value: 36.551
    - type: mrr_at_1000
      value: 36.623
    - type: mrr_at_3
      value: 33.399
    - type: mrr_at_5
      value: 34.575
    - type: ndcg_at_1
      value: 29.644
    - type: ndcg_at_10
      value: 36.521
    - type: ndcg_at_100
      value: 42.087
    - type: ndcg_at_1000
      value: 45.119
    - type: ndcg_at_3
      value: 32.797
    - type: ndcg_at_5
      value: 34.208
    - type: precision_at_1
      value: 29.644
    - type: precision_at_10
      value: 6.7
    - type: precision_at_100
      value: 1.374
    - type: precision_at_1000
      value: 0.22899999999999998
    - type: precision_at_3
      value: 15.152
    - type: precision_at_5
      value: 10.671999999999999
    - type: recall_at_1
      value: 24.855
    - type: recall_at_10
      value: 45.449
    - type: recall_at_100
      value: 70.921
    - type: recall_at_1000
      value: 90.629
    - type: recall_at_3
      value: 33.526
    - type: recall_at_5
      value: 37.848
    - type: map_at_1
      value: 24.781
    - type: map_at_10
      value: 30.020999999999997
    - type: map_at_100
      value: 30.948999999999998
    - type: map_at_1000
      value: 31.05
    - type: map_at_3
      value: 28.412
    - type: map_at_5
      value: 29.193
    - type: mrr_at_1
      value: 27.172
    - type: mrr_at_10
      value: 32.309
    - type: mrr_at_100
      value: 33.164
    - type: mrr_at_1000
      value: 33.239999999999995
    - type: mrr_at_3
      value: 30.775999999999996
    - type: mrr_at_5
      value: 31.562
    - type: ndcg_at_1
      value: 27.172
    - type: ndcg_at_10
      value: 33.178999999999995
    - type: ndcg_at_100
      value: 37.949
    - type: ndcg_at_1000
      value: 40.635
    - type: ndcg_at_3
      value: 30.107
    - type: ndcg_at_5
      value: 31.36
    - type: precision_at_1
      value: 27.172
    - type: precision_at_10
      value: 4.769
    - type: precision_at_100
      value: 0.769
    - type: precision_at_1000
      value: 0.109
    - type: precision_at_3
      value: 12.261
    - type: precision_at_5
      value: 8.17
    - type: recall_at_1
      value: 24.781
    - type: recall_at_10
      value: 40.699000000000005
    - type: recall_at_100
      value: 62.866
    - type: recall_at_1000
      value: 83.11699999999999
    - type: recall_at_3
      value: 32.269999999999996
    - type: recall_at_5
      value: 35.443999999999996
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
      value: 5.2139999999999995
    - type: map_at_10
      value: 9.986
    - type: map_at_100
      value: 11.343
    - type: map_at_1000
      value: 11.55
    - type: map_at_3
      value: 7.961
    - type: map_at_5
      value: 8.967
    - type: mrr_at_1
      value: 12.052
    - type: mrr_at_10
      value: 20.165
    - type: mrr_at_100
      value: 21.317
    - type: mrr_at_1000
      value: 21.399
    - type: mrr_at_3
      value: 17.079
    - type: mrr_at_5
      value: 18.695
    - type: ndcg_at_1
      value: 12.052
    - type: ndcg_at_10
      value: 15.375
    - type: ndcg_at_100
      value: 21.858
    - type: ndcg_at_1000
      value: 26.145000000000003
    - type: ndcg_at_3
      value: 11.334
    - type: ndcg_at_5
      value: 12.798000000000002
    - type: precision_at_1
      value: 12.052
    - type: precision_at_10
      value: 5.16
    - type: precision_at_100
      value: 1.206
    - type: precision_at_1000
      value: 0.198
    - type: precision_at_3
      value: 8.73
    - type: precision_at_5
      value: 7.114
    - type: recall_at_1
      value: 5.2139999999999995
    - type: recall_at_10
      value: 20.669999999999998
    - type: recall_at_100
      value: 43.901
    - type: recall_at_1000
      value: 68.447
    - type: recall_at_3
      value: 11.049000000000001
    - type: recall_at_5
      value: 14.652999999999999
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
      value: 8.511000000000001
    - type: map_at_10
      value: 19.503
    - type: map_at_100
      value: 27.46
    - type: map_at_1000
      value: 29.187
    - type: map_at_3
      value: 14.030999999999999
    - type: map_at_5
      value: 16.329
    - type: mrr_at_1
      value: 63.74999999999999
    - type: mrr_at_10
      value: 73.419
    - type: mrr_at_100
      value: 73.691
    - type: mrr_at_1000
      value: 73.697
    - type: mrr_at_3
      value: 71.792
    - type: mrr_at_5
      value: 72.979
    - type: ndcg_at_1
      value: 53.125
    - type: ndcg_at_10
      value: 41.02
    - type: ndcg_at_100
      value: 45.407
    - type: ndcg_at_1000
      value: 52.68000000000001
    - type: ndcg_at_3
      value: 46.088
    - type: ndcg_at_5
      value: 43.236000000000004
    - type: precision_at_1
      value: 63.74999999999999
    - type: precision_at_10
      value: 32.35
    - type: precision_at_100
      value: 10.363
    - type: precision_at_1000
      value: 2.18
    - type: precision_at_3
      value: 49.667
    - type: precision_at_5
      value: 41.5
    - type: recall_at_1
      value: 8.511000000000001
    - type: recall_at_10
      value: 24.851
    - type: recall_at_100
      value: 50.745
    - type: recall_at_1000
      value: 73.265
    - type: recall_at_3
      value: 15.716
    - type: recall_at_5
      value: 19.256
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
      value: 49.43500000000001
    - type: f1
      value: 44.56288273966374
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
      value: 40.858
    - type: map_at_10
      value: 52.276
    - type: map_at_100
      value: 52.928
    - type: map_at_1000
      value: 52.966
    - type: map_at_3
      value: 49.729
    - type: map_at_5
      value: 51.27
    - type: mrr_at_1
      value: 43.624
    - type: mrr_at_10
      value: 55.22899999999999
    - type: mrr_at_100
      value: 55.823
    - type: mrr_at_1000
      value: 55.85
    - type: mrr_at_3
      value: 52.739999999999995
    - type: mrr_at_5
      value: 54.251000000000005
    - type: ndcg_at_1
      value: 43.624
    - type: ndcg_at_10
      value: 58.23500000000001
    - type: ndcg_at_100
      value: 61.315
    - type: ndcg_at_1000
      value: 62.20099999999999
    - type: ndcg_at_3
      value: 53.22
    - type: ndcg_at_5
      value: 55.88999999999999
    - type: precision_at_1
      value: 43.624
    - type: precision_at_10
      value: 8.068999999999999
    - type: precision_at_100
      value: 0.975
    - type: precision_at_1000
      value: 0.107
    - type: precision_at_3
      value: 21.752
    - type: precision_at_5
      value: 14.515
    - type: recall_at_1
      value: 40.858
    - type: recall_at_10
      value: 73.744
    - type: recall_at_100
      value: 87.667
    - type: recall_at_1000
      value: 94.15599999999999
    - type: recall_at_3
      value: 60.287
    - type: recall_at_5
      value: 66.703
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
      value: 17.864
    - type: map_at_10
      value: 28.592000000000002
    - type: map_at_100
      value: 30.165
    - type: map_at_1000
      value: 30.364
    - type: map_at_3
      value: 24.586
    - type: map_at_5
      value: 26.717000000000002
    - type: mrr_at_1
      value: 35.031
    - type: mrr_at_10
      value: 43.876
    - type: mrr_at_100
      value: 44.683
    - type: mrr_at_1000
      value: 44.736
    - type: mrr_at_3
      value: 40.998000000000005
    - type: mrr_at_5
      value: 42.595
    - type: ndcg_at_1
      value: 35.031
    - type: ndcg_at_10
      value: 36.368
    - type: ndcg_at_100
      value: 42.472
    - type: ndcg_at_1000
      value: 45.973000000000006
    - type: ndcg_at_3
      value: 31.915
    - type: ndcg_at_5
      value: 33.394
    - type: precision_at_1
      value: 35.031
    - type: precision_at_10
      value: 10.139
    - type: precision_at_100
      value: 1.6420000000000001
    - type: precision_at_1000
      value: 0.22699999999999998
    - type: precision_at_3
      value: 21.142
    - type: precision_at_5
      value: 15.772
    - type: recall_at_1
      value: 17.864
    - type: recall_at_10
      value: 43.991
    - type: recall_at_100
      value: 66.796
    - type: recall_at_1000
      value: 87.64
    - type: recall_at_3
      value: 28.915999999999997
    - type: recall_at_5
      value: 35.185
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
      value: 36.556
    - type: map_at_10
      value: 53.056000000000004
    - type: map_at_100
      value: 53.909
    - type: map_at_1000
      value: 53.98
    - type: map_at_3
      value: 49.982
    - type: map_at_5
      value: 51.9
    - type: mrr_at_1
      value: 73.113
    - type: mrr_at_10
      value: 79.381
    - type: mrr_at_100
      value: 79.60300000000001
    - type: mrr_at_1000
      value: 79.617
    - type: mrr_at_3
      value: 78.298
    - type: mrr_at_5
      value: 78.995
    - type: ndcg_at_1
      value: 73.113
    - type: ndcg_at_10
      value: 62.21
    - type: ndcg_at_100
      value: 65.242
    - type: ndcg_at_1000
      value: 66.667
    - type: ndcg_at_3
      value: 57.717
    - type: ndcg_at_5
      value: 60.224
    - type: precision_at_1
      value: 73.113
    - type: precision_at_10
      value: 12.842999999999998
    - type: precision_at_100
      value: 1.522
    - type: precision_at_1000
      value: 0.17099999999999999
    - type: precision_at_3
      value: 36.178
    - type: precision_at_5
      value: 23.695
    - type: recall_at_1
      value: 36.556
    - type: recall_at_10
      value: 64.213
    - type: recall_at_100
      value: 76.077
    - type: recall_at_1000
      value: 85.53699999999999
    - type: recall_at_3
      value: 54.266999999999996
    - type: recall_at_5
      value: 59.236999999999995
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
      value: 75.958
    - type: ap
      value: 69.82869527654348
    - type: f1
      value: 75.89120903005633
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
      value: 23.608
    - type: map_at_10
      value: 36.144
    - type: map_at_100
      value: 37.244
    - type: map_at_1000
      value: 37.291999999999994
    - type: map_at_3
      value: 32.287
    - type: map_at_5
      value: 34.473
    - type: mrr_at_1
      value: 24.226
    - type: mrr_at_10
      value: 36.711
    - type: mrr_at_100
      value: 37.758
    - type: mrr_at_1000
      value: 37.8
    - type: mrr_at_3
      value: 32.92
    - type: mrr_at_5
      value: 35.104
    - type: ndcg_at_1
      value: 24.269
    - type: ndcg_at_10
      value: 43.138
    - type: ndcg_at_100
      value: 48.421
    - type: ndcg_at_1000
      value: 49.592000000000006
    - type: ndcg_at_3
      value: 35.269
    - type: ndcg_at_5
      value: 39.175
    - type: precision_at_1
      value: 24.269
    - type: precision_at_10
      value: 6.755999999999999
    - type: precision_at_100
      value: 0.941
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.938
    - type: precision_at_5
      value: 10.934000000000001
    - type: recall_at_1
      value: 23.608
    - type: recall_at_10
      value: 64.679
    - type: recall_at_100
      value: 89.027
    - type: recall_at_1000
      value: 97.91
    - type: recall_at_3
      value: 43.25
    - type: recall_at_5
      value: 52.617000000000004
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
      value: 93.21477428180576
    - type: f1
      value: 92.92502305092152
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
      value: 74.76744186046511
    - type: f1
      value: 59.19855520057899
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
      value: 72.24613315400134
    - type: f1
      value: 70.19950395651232
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
      value: 76.75857431069268
    - type: f1
      value: 76.5433450230191
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
      value: 31.525463791623604
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
      value: 28.28695907385136
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
      value: 30.068174046665224
    - type: mrr
      value: 30.827586642840803
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
      value: 6.322
    - type: map_at_10
      value: 13.919999999999998
    - type: map_at_100
      value: 17.416
    - type: map_at_1000
      value: 18.836
    - type: map_at_3
      value: 10.111
    - type: map_at_5
      value: 11.991999999999999
    - type: mrr_at_1
      value: 48.297000000000004
    - type: mrr_at_10
      value: 57.114
    - type: mrr_at_100
      value: 57.713
    - type: mrr_at_1000
      value: 57.751
    - type: mrr_at_3
      value: 55.108000000000004
    - type: mrr_at_5
      value: 56.533
    - type: ndcg_at_1
      value: 46.44
    - type: ndcg_at_10
      value: 36.589
    - type: ndcg_at_100
      value: 33.202
    - type: ndcg_at_1000
      value: 41.668
    - type: ndcg_at_3
      value: 41.302
    - type: ndcg_at_5
      value: 39.829
    - type: precision_at_1
      value: 47.988
    - type: precision_at_10
      value: 27.059
    - type: precision_at_100
      value: 8.235000000000001
    - type: precision_at_1000
      value: 2.091
    - type: precision_at_3
      value: 38.184000000000005
    - type: precision_at_5
      value: 34.365
    - type: recall_at_1
      value: 6.322
    - type: recall_at_10
      value: 18.288
    - type: recall_at_100
      value: 32.580999999999996
    - type: recall_at_1000
      value: 63.605999999999995
    - type: recall_at_3
      value: 11.266
    - type: recall_at_5
      value: 14.69
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
      value: 36.586999999999996
    - type: map_at_10
      value: 52.464
    - type: map_at_100
      value: 53.384
    - type: map_at_1000
      value: 53.405
    - type: map_at_3
      value: 48.408
    - type: map_at_5
      value: 50.788999999999994
    - type: mrr_at_1
      value: 40.904
    - type: mrr_at_10
      value: 54.974000000000004
    - type: mrr_at_100
      value: 55.60699999999999
    - type: mrr_at_1000
      value: 55.623
    - type: mrr_at_3
      value: 51.73799999999999
    - type: mrr_at_5
      value: 53.638
    - type: ndcg_at_1
      value: 40.904
    - type: ndcg_at_10
      value: 59.965999999999994
    - type: ndcg_at_100
      value: 63.613
    - type: ndcg_at_1000
      value: 64.064
    - type: ndcg_at_3
      value: 52.486
    - type: ndcg_at_5
      value: 56.377
    - type: precision_at_1
      value: 40.904
    - type: precision_at_10
      value: 9.551
    - type: precision_at_100
      value: 1.162
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 23.552
    - type: precision_at_5
      value: 16.436999999999998
    - type: recall_at_1
      value: 36.586999999999996
    - type: recall_at_10
      value: 80.094
    - type: recall_at_100
      value: 95.515
    - type: recall_at_1000
      value: 98.803
    - type: recall_at_3
      value: 60.907
    - type: recall_at_5
      value: 69.817
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
      value: 70.422
    - type: map_at_10
      value: 84.113
    - type: map_at_100
      value: 84.744
    - type: map_at_1000
      value: 84.762
    - type: map_at_3
      value: 81.171
    - type: map_at_5
      value: 83.039
    - type: mrr_at_1
      value: 81.12
    - type: mrr_at_10
      value: 87.277
    - type: mrr_at_100
      value: 87.384
    - type: mrr_at_1000
      value: 87.385
    - type: mrr_at_3
      value: 86.315
    - type: mrr_at_5
      value: 86.981
    - type: ndcg_at_1
      value: 81.12
    - type: ndcg_at_10
      value: 87.92
    - type: ndcg_at_100
      value: 89.178
    - type: ndcg_at_1000
      value: 89.29899999999999
    - type: ndcg_at_3
      value: 85.076
    - type: ndcg_at_5
      value: 86.67099999999999
    - type: precision_at_1
      value: 81.12
    - type: precision_at_10
      value: 13.325999999999999
    - type: precision_at_100
      value: 1.524
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.16
    - type: precision_at_5
      value: 24.456
    - type: recall_at_1
      value: 70.422
    - type: recall_at_10
      value: 95.00800000000001
    - type: recall_at_100
      value: 99.38
    - type: recall_at_1000
      value: 99.94800000000001
    - type: recall_at_3
      value: 86.809
    - type: recall_at_5
      value: 91.334
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
      value: 48.18491891699636
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
      value: 62.190639679711914
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
      value: 11.268
    - type: map_at_100
      value: 13.129
    - type: map_at_1000
      value: 13.41
    - type: map_at_3
      value: 8.103
    - type: map_at_5
      value: 9.609
    - type: mrr_at_1
      value: 22
    - type: mrr_at_10
      value: 32.248
    - type: mrr_at_100
      value: 33.355000000000004
    - type: mrr_at_1000
      value: 33.42
    - type: mrr_at_3
      value: 29.15
    - type: mrr_at_5
      value: 30.785
    - type: ndcg_at_1
      value: 22
    - type: ndcg_at_10
      value: 18.990000000000002
    - type: ndcg_at_100
      value: 26.302999999999997
    - type: ndcg_at_1000
      value: 31.537
    - type: ndcg_at_3
      value: 18.034
    - type: ndcg_at_5
      value: 15.655
    - type: precision_at_1
      value: 22
    - type: precision_at_10
      value: 9.91
    - type: precision_at_100
      value: 2.0420000000000003
    - type: precision_at_1000
      value: 0.33
    - type: precision_at_3
      value: 16.933
    - type: precision_at_5
      value: 13.719999999999999
    - type: recall_at_1
      value: 4.478
    - type: recall_at_10
      value: 20.087
    - type: recall_at_100
      value: 41.457
    - type: recall_at_1000
      value: 67.10199999999999
    - type: recall_at_3
      value: 10.313
    - type: recall_at_5
      value: 13.927999999999999
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
      value: 84.27341574565806
    - type: cos_sim_spearman
      value: 79.66419880841734
    - type: euclidean_pearson
      value: 81.32473321838208
    - type: euclidean_spearman
      value: 79.29828832085133
    - type: manhattan_pearson
      value: 81.25554065883132
    - type: manhattan_spearman
      value: 79.23275543279853
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
      value: 83.40468875905418
    - type: cos_sim_spearman
      value: 74.2189990321174
    - type: euclidean_pearson
      value: 80.74376966290956
    - type: euclidean_spearman
      value: 74.97663839079335
    - type: manhattan_pearson
      value: 80.69779331646207
    - type: manhattan_spearman
      value: 75.00225252917613
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
      value: 82.5745290053095
    - type: cos_sim_spearman
      value: 83.31401180333397
    - type: euclidean_pearson
      value: 82.96500607325534
    - type: euclidean_spearman
      value: 83.8534967935793
    - type: manhattan_pearson
      value: 82.83112050632508
    - type: manhattan_spearman
      value: 83.70877296557838
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
      value: 80.67833656607704
    - type: cos_sim_spearman
      value: 78.52252410630707
    - type: euclidean_pearson
      value: 80.071189514343
    - type: euclidean_spearman
      value: 78.95143545742796
    - type: manhattan_pearson
      value: 80.0128926165121
    - type: manhattan_spearman
      value: 78.91236678732628
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
      value: 87.48437639980746
    - type: cos_sim_spearman
      value: 88.34876527774259
    - type: euclidean_pearson
      value: 87.64898081823888
    - type: euclidean_spearman
      value: 88.58937180804213
    - type: manhattan_pearson
      value: 87.5942417815288
    - type: manhattan_spearman
      value: 88.53013922267687
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
      value: 82.69189187164781
    - type: cos_sim_spearman
      value: 84.15327883572112
    - type: euclidean_pearson
      value: 83.64202266685898
    - type: euclidean_spearman
      value: 84.6219602318862
    - type: manhattan_pearson
      value: 83.53256698709998
    - type: manhattan_spearman
      value: 84.49260712904946
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
      value: 87.09508017611589
    - type: cos_sim_spearman
      value: 87.23010990417097
    - type: euclidean_pearson
      value: 87.62545569077133
    - type: euclidean_spearman
      value: 86.71152051711714
    - type: manhattan_pearson
      value: 87.5057154278377
    - type: manhattan_spearman
      value: 86.60611898281267
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
      value: 61.72129893941176
    - type: cos_sim_spearman
      value: 62.87871412069194
    - type: euclidean_pearson
      value: 63.21077648290454
    - type: euclidean_spearman
      value: 63.03263080805978
    - type: manhattan_pearson
      value: 63.20740860135976
    - type: manhattan_spearman
      value: 62.89930471802817
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
      value: 85.039118236799
    - type: cos_sim_spearman
      value: 86.18102563389962
    - type: euclidean_pearson
      value: 85.62977041471879
    - type: euclidean_spearman
      value: 86.02478990544347
    - type: manhattan_pearson
      value: 85.60786740521806
    - type: manhattan_spearman
      value: 85.99546210442547
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
      value: 82.89875069737266
    - type: mrr
      value: 95.42621322033087
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
      value: 58.660999999999994
    - type: map_at_10
      value: 68.738
    - type: map_at_100
      value: 69.33200000000001
    - type: map_at_1000
      value: 69.352
    - type: map_at_3
      value: 66.502
    - type: map_at_5
      value: 67.686
    - type: mrr_at_1
      value: 61.667
    - type: mrr_at_10
      value: 70.003
    - type: mrr_at_100
      value: 70.441
    - type: mrr_at_1000
      value: 70.46
    - type: mrr_at_3
      value: 68.278
    - type: mrr_at_5
      value: 69.194
    - type: ndcg_at_1
      value: 61.667
    - type: ndcg_at_10
      value: 73.083
    - type: ndcg_at_100
      value: 75.56
    - type: ndcg_at_1000
      value: 76.01400000000001
    - type: ndcg_at_3
      value: 69.28699999999999
    - type: ndcg_at_5
      value: 70.85000000000001
    - type: precision_at_1
      value: 61.667
    - type: precision_at_10
      value: 9.6
    - type: precision_at_100
      value: 1.087
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 27.111
    - type: precision_at_5
      value: 17.467
    - type: recall_at_1
      value: 58.660999999999994
    - type: recall_at_10
      value: 85.02199999999999
    - type: recall_at_100
      value: 95.933
    - type: recall_at_1000
      value: 99.333
    - type: recall_at_3
      value: 74.506
    - type: recall_at_5
      value: 78.583
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
      value: 99.8029702970297
    - type: cos_sim_ap
      value: 94.87673936635738
    - type: cos_sim_f1
      value: 90.00502260170768
    - type: cos_sim_precision
      value: 90.41372351160445
    - type: cos_sim_recall
      value: 89.60000000000001
    - type: dot_accuracy
      value: 99.57524752475247
    - type: dot_ap
      value: 84.81717934496321
    - type: dot_f1
      value: 78.23026646556059
    - type: dot_precision
      value: 78.66531850353893
    - type: dot_recall
      value: 77.8
    - type: euclidean_accuracy
      value: 99.8029702970297
    - type: euclidean_ap
      value: 94.74658253135284
    - type: euclidean_f1
      value: 90.08470353761834
    - type: euclidean_precision
      value: 89.77159880834161
    - type: euclidean_recall
      value: 90.4
    - type: manhattan_accuracy
      value: 99.8
    - type: manhattan_ap
      value: 94.69224030742787
    - type: manhattan_f1
      value: 89.9502487562189
    - type: manhattan_precision
      value: 89.50495049504951
    - type: manhattan_recall
      value: 90.4
    - type: max_accuracy
      value: 99.8029702970297
    - type: max_ap
      value: 94.87673936635738
    - type: max_f1
      value: 90.08470353761834
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
      value: 63.906039623153035
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
      value: 32.56053830923281
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
      value: 50.15326538775145
    - type: mrr
      value: 50.99279295051355
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
      value: 31.44030762047337
    - type: cos_sim_spearman
      value: 31.00910300264562
    - type: dot_pearson
      value: 26.88257194766013
    - type: dot_spearman
      value: 27.646202679013577
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
      value: 0.247
    - type: map_at_10
      value: 1.9429999999999998
    - type: map_at_100
      value: 10.82
    - type: map_at_1000
      value: 25.972
    - type: map_at_3
      value: 0.653
    - type: map_at_5
      value: 1.057
    - type: mrr_at_1
      value: 94
    - type: mrr_at_10
      value: 96.333
    - type: mrr_at_100
      value: 96.333
    - type: mrr_at_1000
      value: 96.333
    - type: mrr_at_3
      value: 96.333
    - type: mrr_at_5
      value: 96.333
    - type: ndcg_at_1
      value: 89
    - type: ndcg_at_10
      value: 79.63799999999999
    - type: ndcg_at_100
      value: 57.961
    - type: ndcg_at_1000
      value: 50.733
    - type: ndcg_at_3
      value: 84.224
    - type: ndcg_at_5
      value: 82.528
    - type: precision_at_1
      value: 94
    - type: precision_at_10
      value: 84.2
    - type: precision_at_100
      value: 59.36
    - type: precision_at_1000
      value: 22.738
    - type: precision_at_3
      value: 88
    - type: precision_at_5
      value: 86.8
    - type: recall_at_1
      value: 0.247
    - type: recall_at_10
      value: 2.131
    - type: recall_at_100
      value: 14.035
    - type: recall_at_1000
      value: 47.457
    - type: recall_at_3
      value: 0.6779999999999999
    - type: recall_at_5
      value: 1.124
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
      value: 2.603
    - type: map_at_10
      value: 11.667
    - type: map_at_100
      value: 16.474
    - type: map_at_1000
      value: 18.074
    - type: map_at_3
      value: 6.03
    - type: map_at_5
      value: 8.067
    - type: mrr_at_1
      value: 34.694
    - type: mrr_at_10
      value: 51.063
    - type: mrr_at_100
      value: 51.908
    - type: mrr_at_1000
      value: 51.908
    - type: mrr_at_3
      value: 47.959
    - type: mrr_at_5
      value: 49.694
    - type: ndcg_at_1
      value: 32.653
    - type: ndcg_at_10
      value: 28.305000000000003
    - type: ndcg_at_100
      value: 35.311
    - type: ndcg_at_1000
      value: 47.644999999999996
    - type: ndcg_at_3
      value: 32.187
    - type: ndcg_at_5
      value: 29.134999999999998
    - type: precision_at_1
      value: 34.694
    - type: precision_at_10
      value: 26.122
    - type: precision_at_100
      value: 6.755
    - type: precision_at_1000
      value: 1.467
    - type: precision_at_3
      value: 34.694
    - type: precision_at_5
      value: 30.203999999999997
    - type: recall_at_1
      value: 2.603
    - type: recall_at_10
      value: 18.716
    - type: recall_at_100
      value: 42.512
    - type: recall_at_1000
      value: 79.32000000000001
    - type: recall_at_3
      value: 7.59
    - type: recall_at_5
      value: 10.949
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
      value: 74.117
    - type: ap
      value: 15.89357321699319
    - type: f1
      value: 57.14385866369257
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
      value: 61.38370118845502
    - type: f1
      value: 61.67038693866553
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
      value: 42.57754941537969
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
      value: 86.1775049174465
    - type: cos_sim_ap
      value: 74.3994879581554
    - type: cos_sim_f1
      value: 69.32903671308551
    - type: cos_sim_precision
      value: 61.48193508879363
    - type: cos_sim_recall
      value: 79.47229551451187
    - type: dot_accuracy
      value: 81.65345413363534
    - type: dot_ap
      value: 59.690898346685096
    - type: dot_f1
      value: 57.27622826467499
    - type: dot_precision
      value: 51.34965473948525
    - type: dot_recall
      value: 64.74934036939314
    - type: euclidean_accuracy
      value: 86.04637301066937
    - type: euclidean_ap
      value: 74.33009001775268
    - type: euclidean_f1
      value: 69.2458374142997
    - type: euclidean_precision
      value: 64.59570580173595
    - type: euclidean_recall
      value: 74.6174142480211
    - type: manhattan_accuracy
      value: 86.11193896405793
    - type: manhattan_ap
      value: 74.2964140130421
    - type: manhattan_f1
      value: 69.11601528788066
    - type: manhattan_precision
      value: 64.86924323073363
    - type: manhattan_recall
      value: 73.95778364116094
    - type: max_accuracy
      value: 86.1775049174465
    - type: max_ap
      value: 74.3994879581554
    - type: max_f1
      value: 69.32903671308551
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
      value: 89.01501921061823
    - type: cos_sim_ap
      value: 85.97819287477351
    - type: cos_sim_f1
      value: 78.33882858518875
    - type: cos_sim_precision
      value: 75.49446626204926
    - type: cos_sim_recall
      value: 81.40591315060055
    - type: dot_accuracy
      value: 86.47494857763806
    - type: dot_ap
      value: 78.77420360340282
    - type: dot_f1
      value: 73.06433247936238
    - type: dot_precision
      value: 67.92140777983595
    - type: dot_recall
      value: 79.04989220819218
    - type: euclidean_accuracy
      value: 88.7297706368611
    - type: euclidean_ap
      value: 85.61550568529317
    - type: euclidean_f1
      value: 77.84805525263539
    - type: euclidean_precision
      value: 73.73639994491117
    - type: euclidean_recall
      value: 82.44533415460425
    - type: manhattan_accuracy
      value: 88.75111576823068
    - type: manhattan_ap
      value: 85.58701671476263
    - type: manhattan_f1
      value: 77.70169909067856
    - type: manhattan_precision
      value: 73.37666780704755
    - type: manhattan_recall
      value: 82.5685247921158
    - type: max_accuracy
      value: 89.01501921061823
    - type: max_ap
      value: 85.97819287477351
    - type: max_f1
      value: 78.33882858518875
---

## E5-base

**News (May 2023): please switch to [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2), which has better performance and same method of usage.**

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

This model has 12 layers and the embedding size is 768.

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


# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: summit define',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
model = AutoModel.from_pretrained('intfloat/e5-base')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

## Training Details

Please refer to our paper at [https://arxiv.org/pdf/2212.03533.pdf](https://arxiv.org/pdf/2212.03533.pdf).

## Benchmark Evaluation

Check out [unilm/e5](https://github.com/microsoft/unilm/tree/master/e5) to reproduce evaluation results 
on the [BEIR](https://arxiv.org/abs/2104.08663) and [MTEB benchmark](https://arxiv.org/abs/2210.07316).

## Support for Sentence Transformers

Below is an example for usage with sentence_transformers.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/e5-base')
input_texts = [
    'query: how much protein should a female eat',
    'query: summit define',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
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

- Use "query: " prefix for symmetric tasks such as semantic similarity, paraphrase retrieval.

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

This model only works for English texts. Long texts will be truncated to at most 512 tokens.
