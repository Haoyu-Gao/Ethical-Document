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
- name: e5-base-v2
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
      value: 77.77611940298506
    - type: ap
      value: 42.052710266606056
    - type: f1
      value: 72.12040628266567
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
      value: 92.81012500000001
    - type: ap
      value: 89.4213700757244
    - type: f1
      value: 92.8039091197065
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
      value: 46.711999999999996
    - type: f1
      value: 46.11544975436018
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
      value: 23.186
    - type: map_at_10
      value: 36.632999999999996
    - type: map_at_100
      value: 37.842
    - type: map_at_1000
      value: 37.865
    - type: map_at_3
      value: 32.278
    - type: map_at_5
      value: 34.760999999999996
    - type: mrr_at_1
      value: 23.400000000000002
    - type: mrr_at_10
      value: 36.721
    - type: mrr_at_100
      value: 37.937
    - type: mrr_at_1000
      value: 37.96
    - type: mrr_at_3
      value: 32.302
    - type: mrr_at_5
      value: 34.894
    - type: ndcg_at_1
      value: 23.186
    - type: ndcg_at_10
      value: 44.49
    - type: ndcg_at_100
      value: 50.065000000000005
    - type: ndcg_at_1000
      value: 50.629999999999995
    - type: ndcg_at_3
      value: 35.461
    - type: ndcg_at_5
      value: 39.969
    - type: precision_at_1
      value: 23.186
    - type: precision_at_10
      value: 6.97
    - type: precision_at_100
      value: 0.951
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 14.912
    - type: precision_at_5
      value: 11.152
    - type: recall_at_1
      value: 23.186
    - type: recall_at_10
      value: 69.70100000000001
    - type: recall_at_100
      value: 95.092
    - type: recall_at_1000
      value: 99.431
    - type: recall_at_3
      value: 44.737
    - type: recall_at_5
      value: 55.761
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
      value: 46.10312401440185
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
      value: 39.67275326095384
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
      value: 58.97793816337376
    - type: mrr
      value: 72.76832431957087
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
      value: 83.11646947018187
    - type: cos_sim_spearman
      value: 81.40064994975234
    - type: euclidean_pearson
      value: 82.37355689019232
    - type: euclidean_spearman
      value: 81.6777646977348
    - type: manhattan_pearson
      value: 82.61101422716945
    - type: manhattan_spearman
      value: 81.80427360442245
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
      value: 83.52922077922076
    - type: f1
      value: 83.45298679360866
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
      value: 37.495115019668496
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
      value: 32.724792944166765
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
      value: 32.361000000000004
    - type: map_at_10
      value: 43.765
    - type: map_at_100
      value: 45.224
    - type: map_at_1000
      value: 45.35
    - type: map_at_3
      value: 40.353
    - type: map_at_5
      value: 42.195
    - type: mrr_at_1
      value: 40.629
    - type: mrr_at_10
      value: 50.458000000000006
    - type: mrr_at_100
      value: 51.06699999999999
    - type: mrr_at_1000
      value: 51.12
    - type: mrr_at_3
      value: 47.902
    - type: mrr_at_5
      value: 49.447
    - type: ndcg_at_1
      value: 40.629
    - type: ndcg_at_10
      value: 50.376
    - type: ndcg_at_100
      value: 55.065
    - type: ndcg_at_1000
      value: 57.196000000000005
    - type: ndcg_at_3
      value: 45.616
    - type: ndcg_at_5
      value: 47.646
    - type: precision_at_1
      value: 40.629
    - type: precision_at_10
      value: 9.785
    - type: precision_at_100
      value: 1.562
    - type: precision_at_1000
      value: 0.2
    - type: precision_at_3
      value: 22.031
    - type: precision_at_5
      value: 15.737000000000002
    - type: recall_at_1
      value: 32.361000000000004
    - type: recall_at_10
      value: 62.214000000000006
    - type: recall_at_100
      value: 81.464
    - type: recall_at_1000
      value: 95.905
    - type: recall_at_3
      value: 47.5
    - type: recall_at_5
      value: 53.69500000000001
    - type: map_at_1
      value: 27.971
    - type: map_at_10
      value: 37.444
    - type: map_at_100
      value: 38.607
    - type: map_at_1000
      value: 38.737
    - type: map_at_3
      value: 34.504000000000005
    - type: map_at_5
      value: 36.234
    - type: mrr_at_1
      value: 35.35
    - type: mrr_at_10
      value: 43.441
    - type: mrr_at_100
      value: 44.147999999999996
    - type: mrr_at_1000
      value: 44.196000000000005
    - type: mrr_at_3
      value: 41.285
    - type: mrr_at_5
      value: 42.552
    - type: ndcg_at_1
      value: 35.35
    - type: ndcg_at_10
      value: 42.903999999999996
    - type: ndcg_at_100
      value: 47.406
    - type: ndcg_at_1000
      value: 49.588
    - type: ndcg_at_3
      value: 38.778
    - type: ndcg_at_5
      value: 40.788000000000004
    - type: precision_at_1
      value: 35.35
    - type: precision_at_10
      value: 8.083
    - type: precision_at_100
      value: 1.313
    - type: precision_at_1000
      value: 0.18
    - type: precision_at_3
      value: 18.769
    - type: precision_at_5
      value: 13.439
    - type: recall_at_1
      value: 27.971
    - type: recall_at_10
      value: 52.492000000000004
    - type: recall_at_100
      value: 71.642
    - type: recall_at_1000
      value: 85.488
    - type: recall_at_3
      value: 40.1
    - type: recall_at_5
      value: 45.800000000000004
    - type: map_at_1
      value: 39.898
    - type: map_at_10
      value: 51.819
    - type: map_at_100
      value: 52.886
    - type: map_at_1000
      value: 52.941
    - type: map_at_3
      value: 48.619
    - type: map_at_5
      value: 50.493
    - type: mrr_at_1
      value: 45.391999999999996
    - type: mrr_at_10
      value: 55.230000000000004
    - type: mrr_at_100
      value: 55.887
    - type: mrr_at_1000
      value: 55.916
    - type: mrr_at_3
      value: 52.717000000000006
    - type: mrr_at_5
      value: 54.222
    - type: ndcg_at_1
      value: 45.391999999999996
    - type: ndcg_at_10
      value: 57.586999999999996
    - type: ndcg_at_100
      value: 61.745000000000005
    - type: ndcg_at_1000
      value: 62.83800000000001
    - type: ndcg_at_3
      value: 52.207
    - type: ndcg_at_5
      value: 54.925999999999995
    - type: precision_at_1
      value: 45.391999999999996
    - type: precision_at_10
      value: 9.21
    - type: precision_at_100
      value: 1.226
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 23.177
    - type: precision_at_5
      value: 16.038
    - type: recall_at_1
      value: 39.898
    - type: recall_at_10
      value: 71.18900000000001
    - type: recall_at_100
      value: 89.082
    - type: recall_at_1000
      value: 96.865
    - type: recall_at_3
      value: 56.907
    - type: recall_at_5
      value: 63.397999999999996
    - type: map_at_1
      value: 22.706
    - type: map_at_10
      value: 30.818
    - type: map_at_100
      value: 32.038
    - type: map_at_1000
      value: 32.123000000000005
    - type: map_at_3
      value: 28.077
    - type: map_at_5
      value: 29.709999999999997
    - type: mrr_at_1
      value: 24.407
    - type: mrr_at_10
      value: 32.555
    - type: mrr_at_100
      value: 33.692
    - type: mrr_at_1000
      value: 33.751
    - type: mrr_at_3
      value: 29.848999999999997
    - type: mrr_at_5
      value: 31.509999999999998
    - type: ndcg_at_1
      value: 24.407
    - type: ndcg_at_10
      value: 35.624
    - type: ndcg_at_100
      value: 41.454
    - type: ndcg_at_1000
      value: 43.556
    - type: ndcg_at_3
      value: 30.217
    - type: ndcg_at_5
      value: 33.111000000000004
    - type: precision_at_1
      value: 24.407
    - type: precision_at_10
      value: 5.548
    - type: precision_at_100
      value: 0.8869999999999999
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 12.731
    - type: precision_at_5
      value: 9.22
    - type: recall_at_1
      value: 22.706
    - type: recall_at_10
      value: 48.772
    - type: recall_at_100
      value: 75.053
    - type: recall_at_1000
      value: 90.731
    - type: recall_at_3
      value: 34.421
    - type: recall_at_5
      value: 41.427
    - type: map_at_1
      value: 13.424
    - type: map_at_10
      value: 21.09
    - type: map_at_100
      value: 22.264999999999997
    - type: map_at_1000
      value: 22.402
    - type: map_at_3
      value: 18.312
    - type: map_at_5
      value: 19.874
    - type: mrr_at_1
      value: 16.915
    - type: mrr_at_10
      value: 25.258000000000003
    - type: mrr_at_100
      value: 26.228
    - type: mrr_at_1000
      value: 26.31
    - type: mrr_at_3
      value: 22.492
    - type: mrr_at_5
      value: 24.04
    - type: ndcg_at_1
      value: 16.915
    - type: ndcg_at_10
      value: 26.266000000000002
    - type: ndcg_at_100
      value: 32.08
    - type: ndcg_at_1000
      value: 35.086
    - type: ndcg_at_3
      value: 21.049
    - type: ndcg_at_5
      value: 23.508000000000003
    - type: precision_at_1
      value: 16.915
    - type: precision_at_10
      value: 5.1
    - type: precision_at_100
      value: 0.9329999999999999
    - type: precision_at_1000
      value: 0.131
    - type: precision_at_3
      value: 10.282
    - type: precision_at_5
      value: 7.836
    - type: recall_at_1
      value: 13.424
    - type: recall_at_10
      value: 38.179
    - type: recall_at_100
      value: 63.906
    - type: recall_at_1000
      value: 84.933
    - type: recall_at_3
      value: 23.878
    - type: recall_at_5
      value: 30.037999999999997
    - type: map_at_1
      value: 26.154
    - type: map_at_10
      value: 35.912
    - type: map_at_100
      value: 37.211
    - type: map_at_1000
      value: 37.327
    - type: map_at_3
      value: 32.684999999999995
    - type: map_at_5
      value: 34.562
    - type: mrr_at_1
      value: 32.435
    - type: mrr_at_10
      value: 41.411
    - type: mrr_at_100
      value: 42.297000000000004
    - type: mrr_at_1000
      value: 42.345
    - type: mrr_at_3
      value: 38.771
    - type: mrr_at_5
      value: 40.33
    - type: ndcg_at_1
      value: 32.435
    - type: ndcg_at_10
      value: 41.785
    - type: ndcg_at_100
      value: 47.469
    - type: ndcg_at_1000
      value: 49.685
    - type: ndcg_at_3
      value: 36.618
    - type: ndcg_at_5
      value: 39.101
    - type: precision_at_1
      value: 32.435
    - type: precision_at_10
      value: 7.642
    - type: precision_at_100
      value: 1.244
    - type: precision_at_1000
      value: 0.163
    - type: precision_at_3
      value: 17.485
    - type: precision_at_5
      value: 12.57
    - type: recall_at_1
      value: 26.154
    - type: recall_at_10
      value: 54.111
    - type: recall_at_100
      value: 78.348
    - type: recall_at_1000
      value: 92.996
    - type: recall_at_3
      value: 39.189
    - type: recall_at_5
      value: 45.852
    - type: map_at_1
      value: 26.308999999999997
    - type: map_at_10
      value: 35.524
    - type: map_at_100
      value: 36.774
    - type: map_at_1000
      value: 36.891
    - type: map_at_3
      value: 32.561
    - type: map_at_5
      value: 34.034
    - type: mrr_at_1
      value: 31.735000000000003
    - type: mrr_at_10
      value: 40.391
    - type: mrr_at_100
      value: 41.227000000000004
    - type: mrr_at_1000
      value: 41.288000000000004
    - type: mrr_at_3
      value: 37.938
    - type: mrr_at_5
      value: 39.193
    - type: ndcg_at_1
      value: 31.735000000000003
    - type: ndcg_at_10
      value: 41.166000000000004
    - type: ndcg_at_100
      value: 46.702
    - type: ndcg_at_1000
      value: 49.157000000000004
    - type: ndcg_at_3
      value: 36.274
    - type: ndcg_at_5
      value: 38.177
    - type: precision_at_1
      value: 31.735000000000003
    - type: precision_at_10
      value: 7.5569999999999995
    - type: precision_at_100
      value: 1.2109999999999999
    - type: precision_at_1000
      value: 0.16
    - type: precision_at_3
      value: 17.199
    - type: precision_at_5
      value: 12.123000000000001
    - type: recall_at_1
      value: 26.308999999999997
    - type: recall_at_10
      value: 53.083000000000006
    - type: recall_at_100
      value: 76.922
    - type: recall_at_1000
      value: 93.767
    - type: recall_at_3
      value: 39.262
    - type: recall_at_5
      value: 44.413000000000004
    - type: map_at_1
      value: 24.391250000000003
    - type: map_at_10
      value: 33.280166666666666
    - type: map_at_100
      value: 34.49566666666667
    - type: map_at_1000
      value: 34.61533333333333
    - type: map_at_3
      value: 30.52183333333333
    - type: map_at_5
      value: 32.06608333333333
    - type: mrr_at_1
      value: 29.105083333333337
    - type: mrr_at_10
      value: 37.44766666666666
    - type: mrr_at_100
      value: 38.32491666666667
    - type: mrr_at_1000
      value: 38.385666666666665
    - type: mrr_at_3
      value: 35.06883333333333
    - type: mrr_at_5
      value: 36.42066666666667
    - type: ndcg_at_1
      value: 29.105083333333337
    - type: ndcg_at_10
      value: 38.54358333333333
    - type: ndcg_at_100
      value: 43.833583333333344
    - type: ndcg_at_1000
      value: 46.215333333333334
    - type: ndcg_at_3
      value: 33.876
    - type: ndcg_at_5
      value: 36.05208333333333
    - type: precision_at_1
      value: 29.105083333333337
    - type: precision_at_10
      value: 6.823416666666665
    - type: precision_at_100
      value: 1.1270833333333334
    - type: precision_at_1000
      value: 0.15208333333333332
    - type: precision_at_3
      value: 15.696750000000002
    - type: precision_at_5
      value: 11.193499999999998
    - type: recall_at_1
      value: 24.391250000000003
    - type: recall_at_10
      value: 49.98808333333333
    - type: recall_at_100
      value: 73.31616666666666
    - type: recall_at_1000
      value: 89.96291666666667
    - type: recall_at_3
      value: 36.86666666666667
    - type: recall_at_5
      value: 42.54350000000001
    - type: map_at_1
      value: 21.995
    - type: map_at_10
      value: 28.807
    - type: map_at_100
      value: 29.813000000000002
    - type: map_at_1000
      value: 29.903000000000002
    - type: map_at_3
      value: 26.636
    - type: map_at_5
      value: 27.912
    - type: mrr_at_1
      value: 24.847
    - type: mrr_at_10
      value: 31.494
    - type: mrr_at_100
      value: 32.381
    - type: mrr_at_1000
      value: 32.446999999999996
    - type: mrr_at_3
      value: 29.473
    - type: mrr_at_5
      value: 30.7
    - type: ndcg_at_1
      value: 24.847
    - type: ndcg_at_10
      value: 32.818999999999996
    - type: ndcg_at_100
      value: 37.835
    - type: ndcg_at_1000
      value: 40.226
    - type: ndcg_at_3
      value: 28.811999999999998
    - type: ndcg_at_5
      value: 30.875999999999998
    - type: precision_at_1
      value: 24.847
    - type: precision_at_10
      value: 5.244999999999999
    - type: precision_at_100
      value: 0.856
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 12.577
    - type: precision_at_5
      value: 8.895999999999999
    - type: recall_at_1
      value: 21.995
    - type: recall_at_10
      value: 42.479
    - type: recall_at_100
      value: 65.337
    - type: recall_at_1000
      value: 83.23700000000001
    - type: recall_at_3
      value: 31.573
    - type: recall_at_5
      value: 36.684
    - type: map_at_1
      value: 15.751000000000001
    - type: map_at_10
      value: 21.909
    - type: map_at_100
      value: 23.064
    - type: map_at_1000
      value: 23.205000000000002
    - type: map_at_3
      value: 20.138
    - type: map_at_5
      value: 20.973
    - type: mrr_at_1
      value: 19.305
    - type: mrr_at_10
      value: 25.647
    - type: mrr_at_100
      value: 26.659
    - type: mrr_at_1000
      value: 26.748
    - type: mrr_at_3
      value: 23.933
    - type: mrr_at_5
      value: 24.754
    - type: ndcg_at_1
      value: 19.305
    - type: ndcg_at_10
      value: 25.886
    - type: ndcg_at_100
      value: 31.56
    - type: ndcg_at_1000
      value: 34.799
    - type: ndcg_at_3
      value: 22.708000000000002
    - type: ndcg_at_5
      value: 23.838
    - type: precision_at_1
      value: 19.305
    - type: precision_at_10
      value: 4.677
    - type: precision_at_100
      value: 0.895
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 10.771
    - type: precision_at_5
      value: 7.46
    - type: recall_at_1
      value: 15.751000000000001
    - type: recall_at_10
      value: 34.156
    - type: recall_at_100
      value: 59.899
    - type: recall_at_1000
      value: 83.08
    - type: recall_at_3
      value: 24.772
    - type: recall_at_5
      value: 28.009
    - type: map_at_1
      value: 23.34
    - type: map_at_10
      value: 32.383
    - type: map_at_100
      value: 33.629999999999995
    - type: map_at_1000
      value: 33.735
    - type: map_at_3
      value: 29.68
    - type: map_at_5
      value: 31.270999999999997
    - type: mrr_at_1
      value: 27.612
    - type: mrr_at_10
      value: 36.381
    - type: mrr_at_100
      value: 37.351
    - type: mrr_at_1000
      value: 37.411
    - type: mrr_at_3
      value: 33.893
    - type: mrr_at_5
      value: 35.353
    - type: ndcg_at_1
      value: 27.612
    - type: ndcg_at_10
      value: 37.714999999999996
    - type: ndcg_at_100
      value: 43.525000000000006
    - type: ndcg_at_1000
      value: 45.812999999999995
    - type: ndcg_at_3
      value: 32.796
    - type: ndcg_at_5
      value: 35.243
    - type: precision_at_1
      value: 27.612
    - type: precision_at_10
      value: 6.465
    - type: precision_at_100
      value: 1.0619999999999998
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 15.049999999999999
    - type: precision_at_5
      value: 10.764999999999999
    - type: recall_at_1
      value: 23.34
    - type: recall_at_10
      value: 49.856
    - type: recall_at_100
      value: 75.334
    - type: recall_at_1000
      value: 91.156
    - type: recall_at_3
      value: 36.497
    - type: recall_at_5
      value: 42.769
    - type: map_at_1
      value: 25.097
    - type: map_at_10
      value: 34.599999999999994
    - type: map_at_100
      value: 36.174
    - type: map_at_1000
      value: 36.398
    - type: map_at_3
      value: 31.781
    - type: map_at_5
      value: 33.22
    - type: mrr_at_1
      value: 31.225
    - type: mrr_at_10
      value: 39.873
    - type: mrr_at_100
      value: 40.853
    - type: mrr_at_1000
      value: 40.904
    - type: mrr_at_3
      value: 37.681
    - type: mrr_at_5
      value: 38.669
    - type: ndcg_at_1
      value: 31.225
    - type: ndcg_at_10
      value: 40.586
    - type: ndcg_at_100
      value: 46.226
    - type: ndcg_at_1000
      value: 48.788
    - type: ndcg_at_3
      value: 36.258
    - type: ndcg_at_5
      value: 37.848
    - type: precision_at_1
      value: 31.225
    - type: precision_at_10
      value: 7.707999999999999
    - type: precision_at_100
      value: 1.536
    - type: precision_at_1000
      value: 0.242
    - type: precision_at_3
      value: 17.26
    - type: precision_at_5
      value: 12.253
    - type: recall_at_1
      value: 25.097
    - type: recall_at_10
      value: 51.602000000000004
    - type: recall_at_100
      value: 76.854
    - type: recall_at_1000
      value: 93.303
    - type: recall_at_3
      value: 38.68
    - type: recall_at_5
      value: 43.258
    - type: map_at_1
      value: 17.689
    - type: map_at_10
      value: 25.291000000000004
    - type: map_at_100
      value: 26.262
    - type: map_at_1000
      value: 26.372
    - type: map_at_3
      value: 22.916
    - type: map_at_5
      value: 24.315
    - type: mrr_at_1
      value: 19.409000000000002
    - type: mrr_at_10
      value: 27.233
    - type: mrr_at_100
      value: 28.109
    - type: mrr_at_1000
      value: 28.192
    - type: mrr_at_3
      value: 24.892
    - type: mrr_at_5
      value: 26.278000000000002
    - type: ndcg_at_1
      value: 19.409000000000002
    - type: ndcg_at_10
      value: 29.809
    - type: ndcg_at_100
      value: 34.936
    - type: ndcg_at_1000
      value: 37.852000000000004
    - type: ndcg_at_3
      value: 25.179000000000002
    - type: ndcg_at_5
      value: 27.563
    - type: precision_at_1
      value: 19.409000000000002
    - type: precision_at_10
      value: 4.861
    - type: precision_at_100
      value: 0.8
    - type: precision_at_1000
      value: 0.116
    - type: precision_at_3
      value: 11.029
    - type: precision_at_5
      value: 7.985
    - type: recall_at_1
      value: 17.689
    - type: recall_at_10
      value: 41.724
    - type: recall_at_100
      value: 65.95299999999999
    - type: recall_at_1000
      value: 88.094
    - type: recall_at_3
      value: 29.621
    - type: recall_at_5
      value: 35.179
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
      value: 10.581
    - type: map_at_10
      value: 18.944
    - type: map_at_100
      value: 20.812
    - type: map_at_1000
      value: 21.002000000000002
    - type: map_at_3
      value: 15.661
    - type: map_at_5
      value: 17.502000000000002
    - type: mrr_at_1
      value: 23.388
    - type: mrr_at_10
      value: 34.263
    - type: mrr_at_100
      value: 35.364000000000004
    - type: mrr_at_1000
      value: 35.409
    - type: mrr_at_3
      value: 30.586000000000002
    - type: mrr_at_5
      value: 32.928000000000004
    - type: ndcg_at_1
      value: 23.388
    - type: ndcg_at_10
      value: 26.56
    - type: ndcg_at_100
      value: 34.248
    - type: ndcg_at_1000
      value: 37.779
    - type: ndcg_at_3
      value: 21.179000000000002
    - type: ndcg_at_5
      value: 23.504
    - type: precision_at_1
      value: 23.388
    - type: precision_at_10
      value: 8.476
    - type: precision_at_100
      value: 1.672
    - type: precision_at_1000
      value: 0.233
    - type: precision_at_3
      value: 15.852
    - type: precision_at_5
      value: 12.73
    - type: recall_at_1
      value: 10.581
    - type: recall_at_10
      value: 32.512
    - type: recall_at_100
      value: 59.313
    - type: recall_at_1000
      value: 79.25
    - type: recall_at_3
      value: 19.912
    - type: recall_at_5
      value: 25.832
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
      value: 9.35
    - type: map_at_10
      value: 20.134
    - type: map_at_100
      value: 28.975
    - type: map_at_1000
      value: 30.709999999999997
    - type: map_at_3
      value: 14.513000000000002
    - type: map_at_5
      value: 16.671
    - type: mrr_at_1
      value: 69.75
    - type: mrr_at_10
      value: 77.67699999999999
    - type: mrr_at_100
      value: 77.97500000000001
    - type: mrr_at_1000
      value: 77.985
    - type: mrr_at_3
      value: 76.292
    - type: mrr_at_5
      value: 77.179
    - type: ndcg_at_1
      value: 56.49999999999999
    - type: ndcg_at_10
      value: 42.226
    - type: ndcg_at_100
      value: 47.562
    - type: ndcg_at_1000
      value: 54.923
    - type: ndcg_at_3
      value: 46.564
    - type: ndcg_at_5
      value: 43.830000000000005
    - type: precision_at_1
      value: 69.75
    - type: precision_at_10
      value: 33.525
    - type: precision_at_100
      value: 11.035
    - type: precision_at_1000
      value: 2.206
    - type: precision_at_3
      value: 49.75
    - type: precision_at_5
      value: 42
    - type: recall_at_1
      value: 9.35
    - type: recall_at_10
      value: 25.793
    - type: recall_at_100
      value: 54.186
    - type: recall_at_1000
      value: 77.81
    - type: recall_at_3
      value: 15.770000000000001
    - type: recall_at_5
      value: 19.09
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
      value: 46.945
    - type: f1
      value: 42.07407842992542
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
      value: 71.04599999999999
    - type: map_at_10
      value: 80.718
    - type: map_at_100
      value: 80.961
    - type: map_at_1000
      value: 80.974
    - type: map_at_3
      value: 79.49199999999999
    - type: map_at_5
      value: 80.32000000000001
    - type: mrr_at_1
      value: 76.388
    - type: mrr_at_10
      value: 85.214
    - type: mrr_at_100
      value: 85.302
    - type: mrr_at_1000
      value: 85.302
    - type: mrr_at_3
      value: 84.373
    - type: mrr_at_5
      value: 84.979
    - type: ndcg_at_1
      value: 76.388
    - type: ndcg_at_10
      value: 84.987
    - type: ndcg_at_100
      value: 85.835
    - type: ndcg_at_1000
      value: 86.04899999999999
    - type: ndcg_at_3
      value: 83.04
    - type: ndcg_at_5
      value: 84.22500000000001
    - type: precision_at_1
      value: 76.388
    - type: precision_at_10
      value: 10.35
    - type: precision_at_100
      value: 1.099
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 32.108
    - type: precision_at_5
      value: 20.033
    - type: recall_at_1
      value: 71.04599999999999
    - type: recall_at_10
      value: 93.547
    - type: recall_at_100
      value: 96.887
    - type: recall_at_1000
      value: 98.158
    - type: recall_at_3
      value: 88.346
    - type: recall_at_5
      value: 91.321
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
      value: 19.8
    - type: map_at_10
      value: 31.979999999999997
    - type: map_at_100
      value: 33.876
    - type: map_at_1000
      value: 34.056999999999995
    - type: map_at_3
      value: 28.067999999999998
    - type: map_at_5
      value: 30.066
    - type: mrr_at_1
      value: 38.735
    - type: mrr_at_10
      value: 47.749
    - type: mrr_at_100
      value: 48.605
    - type: mrr_at_1000
      value: 48.644999999999996
    - type: mrr_at_3
      value: 45.165
    - type: mrr_at_5
      value: 46.646
    - type: ndcg_at_1
      value: 38.735
    - type: ndcg_at_10
      value: 39.883
    - type: ndcg_at_100
      value: 46.983000000000004
    - type: ndcg_at_1000
      value: 50.043000000000006
    - type: ndcg_at_3
      value: 35.943000000000005
    - type: ndcg_at_5
      value: 37.119
    - type: precision_at_1
      value: 38.735
    - type: precision_at_10
      value: 10.940999999999999
    - type: precision_at_100
      value: 1.836
    - type: precision_at_1000
      value: 0.23900000000000002
    - type: precision_at_3
      value: 23.817
    - type: precision_at_5
      value: 17.346
    - type: recall_at_1
      value: 19.8
    - type: recall_at_10
      value: 47.082
    - type: recall_at_100
      value: 73.247
    - type: recall_at_1000
      value: 91.633
    - type: recall_at_3
      value: 33.201
    - type: recall_at_5
      value: 38.81
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
      value: 38.102999999999994
    - type: map_at_10
      value: 60.547
    - type: map_at_100
      value: 61.466
    - type: map_at_1000
      value: 61.526
    - type: map_at_3
      value: 56.973
    - type: map_at_5
      value: 59.244
    - type: mrr_at_1
      value: 76.205
    - type: mrr_at_10
      value: 82.816
    - type: mrr_at_100
      value: 83.002
    - type: mrr_at_1000
      value: 83.009
    - type: mrr_at_3
      value: 81.747
    - type: mrr_at_5
      value: 82.467
    - type: ndcg_at_1
      value: 76.205
    - type: ndcg_at_10
      value: 69.15
    - type: ndcg_at_100
      value: 72.297
    - type: ndcg_at_1000
      value: 73.443
    - type: ndcg_at_3
      value: 64.07000000000001
    - type: ndcg_at_5
      value: 66.96600000000001
    - type: precision_at_1
      value: 76.205
    - type: precision_at_10
      value: 14.601
    - type: precision_at_100
      value: 1.7049999999999998
    - type: precision_at_1000
      value: 0.186
    - type: precision_at_3
      value: 41.202
    - type: precision_at_5
      value: 27.006000000000004
    - type: recall_at_1
      value: 38.102999999999994
    - type: recall_at_10
      value: 73.005
    - type: recall_at_100
      value: 85.253
    - type: recall_at_1000
      value: 92.795
    - type: recall_at_3
      value: 61.803
    - type: recall_at_5
      value: 67.515
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
      value: 86.15
    - type: ap
      value: 80.36282825265391
    - type: f1
      value: 86.07368510726472
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
      value: 22.6
    - type: map_at_10
      value: 34.887
    - type: map_at_100
      value: 36.069
    - type: map_at_1000
      value: 36.115
    - type: map_at_3
      value: 31.067
    - type: map_at_5
      value: 33.300000000000004
    - type: mrr_at_1
      value: 23.238
    - type: mrr_at_10
      value: 35.47
    - type: mrr_at_100
      value: 36.599
    - type: mrr_at_1000
      value: 36.64
    - type: mrr_at_3
      value: 31.735999999999997
    - type: mrr_at_5
      value: 33.939
    - type: ndcg_at_1
      value: 23.252
    - type: ndcg_at_10
      value: 41.765
    - type: ndcg_at_100
      value: 47.402
    - type: ndcg_at_1000
      value: 48.562
    - type: ndcg_at_3
      value: 34.016999999999996
    - type: ndcg_at_5
      value: 38.016
    - type: precision_at_1
      value: 23.252
    - type: precision_at_10
      value: 6.569
    - type: precision_at_100
      value: 0.938
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.479000000000001
    - type: precision_at_5
      value: 10.722
    - type: recall_at_1
      value: 22.6
    - type: recall_at_10
      value: 62.919000000000004
    - type: recall_at_100
      value: 88.82
    - type: recall_at_1000
      value: 97.71600000000001
    - type: recall_at_3
      value: 41.896
    - type: recall_at_5
      value: 51.537
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
      value: 93.69357045143639
    - type: f1
      value: 93.55489858177597
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
      value: 75.31235750114
    - type: f1
      value: 57.891491963121155
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
      value: 73.04303967720243
    - type: f1
      value: 70.51516022297616
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
      value: 77.65299260255549
    - type: f1
      value: 77.49059766538576
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
      value: 31.458906115906597
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
      value: 28.9851513122443
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
      value: 31.2916268497217
    - type: mrr
      value: 32.328276715593816
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
      value: 6.3740000000000006
    - type: map_at_10
      value: 13.089999999999998
    - type: map_at_100
      value: 16.512
    - type: map_at_1000
      value: 18.014
    - type: map_at_3
      value: 9.671000000000001
    - type: map_at_5
      value: 11.199
    - type: mrr_at_1
      value: 46.749
    - type: mrr_at_10
      value: 55.367
    - type: mrr_at_100
      value: 56.021
    - type: mrr_at_1000
      value: 56.058
    - type: mrr_at_3
      value: 53.30200000000001
    - type: mrr_at_5
      value: 54.773
    - type: ndcg_at_1
      value: 45.046
    - type: ndcg_at_10
      value: 35.388999999999996
    - type: ndcg_at_100
      value: 32.175
    - type: ndcg_at_1000
      value: 41.018
    - type: ndcg_at_3
      value: 40.244
    - type: ndcg_at_5
      value: 38.267
    - type: precision_at_1
      value: 46.749
    - type: precision_at_10
      value: 26.563
    - type: precision_at_100
      value: 8.074
    - type: precision_at_1000
      value: 2.099
    - type: precision_at_3
      value: 37.358000000000004
    - type: precision_at_5
      value: 33.003
    - type: recall_at_1
      value: 6.3740000000000006
    - type: recall_at_10
      value: 16.805999999999997
    - type: recall_at_100
      value: 31.871
    - type: recall_at_1000
      value: 64.098
    - type: recall_at_3
      value: 10.383000000000001
    - type: recall_at_5
      value: 13.166
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
      value: 34.847
    - type: map_at_10
      value: 50.532
    - type: map_at_100
      value: 51.504000000000005
    - type: map_at_1000
      value: 51.528
    - type: map_at_3
      value: 46.219
    - type: map_at_5
      value: 48.868
    - type: mrr_at_1
      value: 39.137
    - type: mrr_at_10
      value: 53.157
    - type: mrr_at_100
      value: 53.839999999999996
    - type: mrr_at_1000
      value: 53.857
    - type: mrr_at_3
      value: 49.667
    - type: mrr_at_5
      value: 51.847
    - type: ndcg_at_1
      value: 39.108
    - type: ndcg_at_10
      value: 58.221000000000004
    - type: ndcg_at_100
      value: 62.021
    - type: ndcg_at_1000
      value: 62.57
    - type: ndcg_at_3
      value: 50.27199999999999
    - type: ndcg_at_5
      value: 54.623999999999995
    - type: precision_at_1
      value: 39.108
    - type: precision_at_10
      value: 9.397
    - type: precision_at_100
      value: 1.1520000000000001
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 22.644000000000002
    - type: precision_at_5
      value: 16.141
    - type: recall_at_1
      value: 34.847
    - type: recall_at_10
      value: 78.945
    - type: recall_at_100
      value: 94.793
    - type: recall_at_1000
      value: 98.904
    - type: recall_at_3
      value: 58.56
    - type: recall_at_5
      value: 68.535
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
      value: 68.728
    - type: map_at_10
      value: 82.537
    - type: map_at_100
      value: 83.218
    - type: map_at_1000
      value: 83.238
    - type: map_at_3
      value: 79.586
    - type: map_at_5
      value: 81.416
    - type: mrr_at_1
      value: 79.17999999999999
    - type: mrr_at_10
      value: 85.79299999999999
    - type: mrr_at_100
      value: 85.937
    - type: mrr_at_1000
      value: 85.938
    - type: mrr_at_3
      value: 84.748
    - type: mrr_at_5
      value: 85.431
    - type: ndcg_at_1
      value: 79.17
    - type: ndcg_at_10
      value: 86.555
    - type: ndcg_at_100
      value: 88.005
    - type: ndcg_at_1000
      value: 88.146
    - type: ndcg_at_3
      value: 83.557
    - type: ndcg_at_5
      value: 85.152
    - type: precision_at_1
      value: 79.17
    - type: precision_at_10
      value: 13.163
    - type: precision_at_100
      value: 1.52
    - type: precision_at_1000
      value: 0.156
    - type: precision_at_3
      value: 36.53
    - type: precision_at_5
      value: 24.046
    - type: recall_at_1
      value: 68.728
    - type: recall_at_10
      value: 94.217
    - type: recall_at_100
      value: 99.295
    - type: recall_at_1000
      value: 99.964
    - type: recall_at_3
      value: 85.646
    - type: recall_at_5
      value: 90.113
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
      value: 56.15680266226348
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
      value: 63.4318549229047
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
      value: 4.353
    - type: map_at_10
      value: 10.956000000000001
    - type: map_at_100
      value: 12.873999999999999
    - type: map_at_1000
      value: 13.177
    - type: map_at_3
      value: 7.854
    - type: map_at_5
      value: 9.327
    - type: mrr_at_1
      value: 21.4
    - type: mrr_at_10
      value: 31.948999999999998
    - type: mrr_at_100
      value: 33.039
    - type: mrr_at_1000
      value: 33.106
    - type: mrr_at_3
      value: 28.449999999999996
    - type: mrr_at_5
      value: 30.535
    - type: ndcg_at_1
      value: 21.4
    - type: ndcg_at_10
      value: 18.694
    - type: ndcg_at_100
      value: 26.275
    - type: ndcg_at_1000
      value: 31.836
    - type: ndcg_at_3
      value: 17.559
    - type: ndcg_at_5
      value: 15.372
    - type: precision_at_1
      value: 21.4
    - type: precision_at_10
      value: 9.790000000000001
    - type: precision_at_100
      value: 2.0709999999999997
    - type: precision_at_1000
      value: 0.34099999999999997
    - type: precision_at_3
      value: 16.467000000000002
    - type: precision_at_5
      value: 13.54
    - type: recall_at_1
      value: 4.353
    - type: recall_at_10
      value: 19.892000000000003
    - type: recall_at_100
      value: 42.067
    - type: recall_at_1000
      value: 69.268
    - type: recall_at_3
      value: 10.042
    - type: recall_at_5
      value: 13.741999999999999
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
      value: 83.75433886279843
    - type: cos_sim_spearman
      value: 78.29727771767095
    - type: euclidean_pearson
      value: 80.83057828506621
    - type: euclidean_spearman
      value: 78.35203149750356
    - type: manhattan_pearson
      value: 80.7403553891142
    - type: manhattan_spearman
      value: 78.33670488531051
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
      value: 84.59999465280839
    - type: cos_sim_spearman
      value: 75.79279003980383
    - type: euclidean_pearson
      value: 82.29895375956758
    - type: euclidean_spearman
      value: 77.33856514102094
    - type: manhattan_pearson
      value: 82.22694214534756
    - type: manhattan_spearman
      value: 77.3028993008695
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
      value: 83.09296929691297
    - type: cos_sim_spearman
      value: 83.58056936846941
    - type: euclidean_pearson
      value: 83.84067483060005
    - type: euclidean_spearman
      value: 84.45155680480985
    - type: manhattan_pearson
      value: 83.82353052971942
    - type: manhattan_spearman
      value: 84.43030567861112
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
      value: 82.74616852320915
    - type: cos_sim_spearman
      value: 79.948683747966
    - type: euclidean_pearson
      value: 81.55702283757084
    - type: euclidean_spearman
      value: 80.1721505114231
    - type: manhattan_pearson
      value: 81.52251518619441
    - type: manhattan_spearman
      value: 80.1469800135577
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
      value: 87.97170104226318
    - type: cos_sim_spearman
      value: 88.82021731518206
    - type: euclidean_pearson
      value: 87.92950547187615
    - type: euclidean_spearman
      value: 88.67043634645866
    - type: manhattan_pearson
      value: 87.90668112827639
    - type: manhattan_spearman
      value: 88.64471082785317
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
      value: 83.02790375770599
    - type: cos_sim_spearman
      value: 84.46308496590792
    - type: euclidean_pearson
      value: 84.29430000414911
    - type: euclidean_spearman
      value: 84.77298303589936
    - type: manhattan_pearson
      value: 84.23919291368665
    - type: manhattan_spearman
      value: 84.75272234871308
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
      value: 87.62885108477064
    - type: cos_sim_spearman
      value: 87.58456196391622
    - type: euclidean_pearson
      value: 88.2602775281007
    - type: euclidean_spearman
      value: 87.51556278299846
    - type: manhattan_pearson
      value: 88.11224053672842
    - type: manhattan_spearman
      value: 87.4336094383095
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
      value: 63.98187965128411
    - type: cos_sim_spearman
      value: 64.0653163219731
    - type: euclidean_pearson
      value: 62.30616725924099
    - type: euclidean_spearman
      value: 61.556971332295916
    - type: manhattan_pearson
      value: 62.07642330128549
    - type: manhattan_spearman
      value: 61.155494129828
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
      value: 85.6089703921826
    - type: cos_sim_spearman
      value: 86.52303197250791
    - type: euclidean_pearson
      value: 85.95801955963246
    - type: euclidean_spearman
      value: 86.25242424112962
    - type: manhattan_pearson
      value: 85.88829100470312
    - type: manhattan_spearman
      value: 86.18742955805165
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
      value: 83.02282098487036
    - type: mrr
      value: 95.05126409538174
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
      value: 55.928
    - type: map_at_10
      value: 67.308
    - type: map_at_100
      value: 67.89500000000001
    - type: map_at_1000
      value: 67.91199999999999
    - type: map_at_3
      value: 65.091
    - type: map_at_5
      value: 66.412
    - type: mrr_at_1
      value: 58.667
    - type: mrr_at_10
      value: 68.401
    - type: mrr_at_100
      value: 68.804
    - type: mrr_at_1000
      value: 68.819
    - type: mrr_at_3
      value: 66.72200000000001
    - type: mrr_at_5
      value: 67.72200000000001
    - type: ndcg_at_1
      value: 58.667
    - type: ndcg_at_10
      value: 71.944
    - type: ndcg_at_100
      value: 74.464
    - type: ndcg_at_1000
      value: 74.82799999999999
    - type: ndcg_at_3
      value: 68.257
    - type: ndcg_at_5
      value: 70.10300000000001
    - type: precision_at_1
      value: 58.667
    - type: precision_at_10
      value: 9.533
    - type: precision_at_100
      value: 1.09
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 27.222
    - type: precision_at_5
      value: 17.533
    - type: recall_at_1
      value: 55.928
    - type: recall_at_10
      value: 84.65
    - type: recall_at_100
      value: 96.267
    - type: recall_at_1000
      value: 99
    - type: recall_at_3
      value: 74.656
    - type: recall_at_5
      value: 79.489
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
      value: 99.79009900990098
    - type: cos_sim_ap
      value: 94.5795129511524
    - type: cos_sim_f1
      value: 89.34673366834171
    - type: cos_sim_precision
      value: 89.79797979797979
    - type: cos_sim_recall
      value: 88.9
    - type: dot_accuracy
      value: 99.53465346534654
    - type: dot_ap
      value: 81.56492504352725
    - type: dot_f1
      value: 76.33816908454227
    - type: dot_precision
      value: 76.37637637637637
    - type: dot_recall
      value: 76.3
    - type: euclidean_accuracy
      value: 99.78514851485149
    - type: euclidean_ap
      value: 94.59134620408962
    - type: euclidean_f1
      value: 88.96484375
    - type: euclidean_precision
      value: 86.92748091603053
    - type: euclidean_recall
      value: 91.10000000000001
    - type: manhattan_accuracy
      value: 99.78415841584159
    - type: manhattan_ap
      value: 94.5190197328845
    - type: manhattan_f1
      value: 88.84462151394423
    - type: manhattan_precision
      value: 88.4920634920635
    - type: manhattan_recall
      value: 89.2
    - type: max_accuracy
      value: 99.79009900990098
    - type: max_ap
      value: 94.59134620408962
    - type: max_f1
      value: 89.34673366834171
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
      value: 65.1487505617497
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
      value: 32.502518166001856
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
      value: 50.33775480236701
    - type: mrr
      value: 51.17302223919871
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
      value: 30.561111309808208
    - type: cos_sim_spearman
      value: 30.2839254379273
    - type: dot_pearson
      value: 29.560242291401973
    - type: dot_spearman
      value: 30.51527274679116
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
      value: 0.215
    - type: map_at_10
      value: 1.752
    - type: map_at_100
      value: 9.258
    - type: map_at_1000
      value: 23.438
    - type: map_at_3
      value: 0.6
    - type: map_at_5
      value: 0.968
    - type: mrr_at_1
      value: 84
    - type: mrr_at_10
      value: 91.333
    - type: mrr_at_100
      value: 91.333
    - type: mrr_at_1000
      value: 91.333
    - type: mrr_at_3
      value: 91.333
    - type: mrr_at_5
      value: 91.333
    - type: ndcg_at_1
      value: 75
    - type: ndcg_at_10
      value: 69.596
    - type: ndcg_at_100
      value: 51.970000000000006
    - type: ndcg_at_1000
      value: 48.864999999999995
    - type: ndcg_at_3
      value: 73.92699999999999
    - type: ndcg_at_5
      value: 73.175
    - type: precision_at_1
      value: 84
    - type: precision_at_10
      value: 74
    - type: precision_at_100
      value: 53.2
    - type: precision_at_1000
      value: 21.836
    - type: precision_at_3
      value: 79.333
    - type: precision_at_5
      value: 78.4
    - type: recall_at_1
      value: 0.215
    - type: recall_at_10
      value: 1.9609999999999999
    - type: recall_at_100
      value: 12.809999999999999
    - type: recall_at_1000
      value: 46.418
    - type: recall_at_3
      value: 0.6479999999999999
    - type: recall_at_5
      value: 1.057
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
      value: 3.066
    - type: map_at_10
      value: 10.508000000000001
    - type: map_at_100
      value: 16.258
    - type: map_at_1000
      value: 17.705000000000002
    - type: map_at_3
      value: 6.157
    - type: map_at_5
      value: 7.510999999999999
    - type: mrr_at_1
      value: 34.694
    - type: mrr_at_10
      value: 48.786
    - type: mrr_at_100
      value: 49.619
    - type: mrr_at_1000
      value: 49.619
    - type: mrr_at_3
      value: 45.918
    - type: mrr_at_5
      value: 46.837
    - type: ndcg_at_1
      value: 31.633
    - type: ndcg_at_10
      value: 26.401999999999997
    - type: ndcg_at_100
      value: 37.139
    - type: ndcg_at_1000
      value: 48.012
    - type: ndcg_at_3
      value: 31.875999999999998
    - type: ndcg_at_5
      value: 27.383000000000003
    - type: precision_at_1
      value: 34.694
    - type: precision_at_10
      value: 22.857
    - type: precision_at_100
      value: 7.611999999999999
    - type: precision_at_1000
      value: 1.492
    - type: precision_at_3
      value: 33.333
    - type: precision_at_5
      value: 26.122
    - type: recall_at_1
      value: 3.066
    - type: recall_at_10
      value: 16.239
    - type: recall_at_100
      value: 47.29
    - type: recall_at_1000
      value: 81.137
    - type: recall_at_3
      value: 7.069
    - type: recall_at_5
      value: 9.483
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
      value: 72.1126
    - type: ap
      value: 14.710862719285753
    - type: f1
      value: 55.437808972378846
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
      value: 60.39049235993209
    - type: f1
      value: 60.69810537250234
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
      value: 48.15576640316866
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
      value: 86.52917684925792
    - type: cos_sim_ap
      value: 75.97497873817315
    - type: cos_sim_f1
      value: 70.01151926276718
    - type: cos_sim_precision
      value: 67.98409147402435
    - type: cos_sim_recall
      value: 72.16358839050132
    - type: dot_accuracy
      value: 82.47004828038385
    - type: dot_ap
      value: 62.48739894974198
    - type: dot_f1
      value: 59.13107511045656
    - type: dot_precision
      value: 55.27765029830197
    - type: dot_recall
      value: 63.562005277044854
    - type: euclidean_accuracy
      value: 86.46361089586935
    - type: euclidean_ap
      value: 75.59282886839452
    - type: euclidean_f1
      value: 69.6465443945099
    - type: euclidean_precision
      value: 64.52847175331982
    - type: euclidean_recall
      value: 75.64643799472296
    - type: manhattan_accuracy
      value: 86.43380818978363
    - type: manhattan_ap
      value: 75.5742420974403
    - type: manhattan_f1
      value: 69.8636926889715
    - type: manhattan_precision
      value: 65.8644859813084
    - type: manhattan_recall
      value: 74.37994722955145
    - type: max_accuracy
      value: 86.52917684925792
    - type: max_ap
      value: 75.97497873817315
    - type: max_f1
      value: 70.01151926276718
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
      value: 89.29056545193464
    - type: cos_sim_ap
      value: 86.63028865482376
    - type: cos_sim_f1
      value: 79.18166458532285
    - type: cos_sim_precision
      value: 75.70585756426465
    - type: cos_sim_recall
      value: 82.99199260856174
    - type: dot_accuracy
      value: 85.23305002522606
    - type: dot_ap
      value: 76.0482687263196
    - type: dot_f1
      value: 70.80484330484332
    - type: dot_precision
      value: 65.86933474688577
    - type: dot_recall
      value: 76.53988296889437
    - type: euclidean_accuracy
      value: 89.26145845461248
    - type: euclidean_ap
      value: 86.54073288416006
    - type: euclidean_f1
      value: 78.9721371479794
    - type: euclidean_precision
      value: 76.68649354417525
    - type: euclidean_recall
      value: 81.39821373575609
    - type: manhattan_accuracy
      value: 89.22847052431405
    - type: manhattan_ap
      value: 86.51250729037905
    - type: manhattan_f1
      value: 78.94601825044894
    - type: manhattan_precision
      value: 75.32694594027555
    - type: manhattan_recall
      value: 82.93039728980598
    - type: max_accuracy
      value: 89.29056545193464
    - type: max_ap
      value: 86.63028865482376
    - type: max_f1
      value: 79.18166458532285
---

# E5-base-v2

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

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = AutoModel.from_pretrained('intfloat/e5-base-v2')

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
model = SentenceTransformer('intfloat/e5-base-v2')
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
