---
language:
- en
license: mit
tags:
- mteb
- sentence-similarity
- sentence-transformers
- Sentence Transformers
model-index:
- name: gte-small
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
      value: 73.22388059701493
    - type: ap
      value: 36.09895941426988
    - type: f1
      value: 67.3205651539195
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
      value: 91.81894999999999
    - type: ap
      value: 88.5240138417305
    - type: f1
      value: 91.80367382706962
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
      value: 48.032
    - type: f1
      value: 47.4490665674719
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
      value: 30.725
    - type: map_at_10
      value: 46.604
    - type: map_at_100
      value: 47.535
    - type: map_at_1000
      value: 47.538000000000004
    - type: map_at_3
      value: 41.833
    - type: map_at_5
      value: 44.61
    - type: mrr_at_1
      value: 31.223
    - type: mrr_at_10
      value: 46.794000000000004
    - type: mrr_at_100
      value: 47.725
    - type: mrr_at_1000
      value: 47.727000000000004
    - type: mrr_at_3
      value: 42.07
    - type: mrr_at_5
      value: 44.812000000000005
    - type: ndcg_at_1
      value: 30.725
    - type: ndcg_at_10
      value: 55.440999999999995
    - type: ndcg_at_100
      value: 59.134
    - type: ndcg_at_1000
      value: 59.199
    - type: ndcg_at_3
      value: 45.599000000000004
    - type: ndcg_at_5
      value: 50.637
    - type: precision_at_1
      value: 30.725
    - type: precision_at_10
      value: 8.364
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 18.848000000000003
    - type: precision_at_5
      value: 13.77
    - type: recall_at_1
      value: 30.725
    - type: recall_at_10
      value: 83.64200000000001
    - type: recall_at_100
      value: 99.14699999999999
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 56.543
    - type: recall_at_5
      value: 68.848
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
      value: 47.90178078197678
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
      value: 40.25728393431922
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
      value: 61.720297062897764
    - type: mrr
      value: 75.24139295607439
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
      value: 89.43527309184616
    - type: cos_sim_spearman
      value: 88.17128615100206
    - type: euclidean_pearson
      value: 87.89922623089282
    - type: euclidean_spearman
      value: 87.96104039655451
    - type: manhattan_pearson
      value: 87.9818290932077
    - type: manhattan_spearman
      value: 88.00923426576885
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
      value: 84.0844155844156
    - type: f1
      value: 84.01485017302213
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
      value: 38.36574769259432
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
      value: 35.4857033165287
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
      value: 30.261
    - type: map_at_10
      value: 42.419000000000004
    - type: map_at_100
      value: 43.927
    - type: map_at_1000
      value: 44.055
    - type: map_at_3
      value: 38.597
    - type: map_at_5
      value: 40.701
    - type: mrr_at_1
      value: 36.91
    - type: mrr_at_10
      value: 48.02
    - type: mrr_at_100
      value: 48.658
    - type: mrr_at_1000
      value: 48.708
    - type: mrr_at_3
      value: 44.945
    - type: mrr_at_5
      value: 46.705000000000005
    - type: ndcg_at_1
      value: 36.91
    - type: ndcg_at_10
      value: 49.353
    - type: ndcg_at_100
      value: 54.456
    - type: ndcg_at_1000
      value: 56.363
    - type: ndcg_at_3
      value: 43.483
    - type: ndcg_at_5
      value: 46.150999999999996
    - type: precision_at_1
      value: 36.91
    - type: precision_at_10
      value: 9.700000000000001
    - type: precision_at_100
      value: 1.557
    - type: precision_at_1000
      value: 0.202
    - type: precision_at_3
      value: 21.078
    - type: precision_at_5
      value: 15.421999999999999
    - type: recall_at_1
      value: 30.261
    - type: recall_at_10
      value: 63.242
    - type: recall_at_100
      value: 84.09100000000001
    - type: recall_at_1000
      value: 96.143
    - type: recall_at_3
      value: 46.478
    - type: recall_at_5
      value: 53.708
    - type: map_at_1
      value: 31.145
    - type: map_at_10
      value: 40.996
    - type: map_at_100
      value: 42.266999999999996
    - type: map_at_1000
      value: 42.397
    - type: map_at_3
      value: 38.005
    - type: map_at_5
      value: 39.628
    - type: mrr_at_1
      value: 38.344
    - type: mrr_at_10
      value: 46.827000000000005
    - type: mrr_at_100
      value: 47.446
    - type: mrr_at_1000
      value: 47.489
    - type: mrr_at_3
      value: 44.448
    - type: mrr_at_5
      value: 45.747
    - type: ndcg_at_1
      value: 38.344
    - type: ndcg_at_10
      value: 46.733000000000004
    - type: ndcg_at_100
      value: 51.103
    - type: ndcg_at_1000
      value: 53.075
    - type: ndcg_at_3
      value: 42.366
    - type: ndcg_at_5
      value: 44.242
    - type: precision_at_1
      value: 38.344
    - type: precision_at_10
      value: 8.822000000000001
    - type: precision_at_100
      value: 1.417
    - type: precision_at_1000
      value: 0.187
    - type: precision_at_3
      value: 20.403
    - type: precision_at_5
      value: 14.306
    - type: recall_at_1
      value: 31.145
    - type: recall_at_10
      value: 56.909
    - type: recall_at_100
      value: 75.274
    - type: recall_at_1000
      value: 87.629
    - type: recall_at_3
      value: 43.784
    - type: recall_at_5
      value: 49.338
    - type: map_at_1
      value: 38.83
    - type: map_at_10
      value: 51.553000000000004
    - type: map_at_100
      value: 52.581
    - type: map_at_1000
      value: 52.638
    - type: map_at_3
      value: 48.112
    - type: map_at_5
      value: 50.095
    - type: mrr_at_1
      value: 44.513999999999996
    - type: mrr_at_10
      value: 54.998000000000005
    - type: mrr_at_100
      value: 55.650999999999996
    - type: mrr_at_1000
      value: 55.679
    - type: mrr_at_3
      value: 52.602000000000004
    - type: mrr_at_5
      value: 53.931
    - type: ndcg_at_1
      value: 44.513999999999996
    - type: ndcg_at_10
      value: 57.67400000000001
    - type: ndcg_at_100
      value: 61.663999999999994
    - type: ndcg_at_1000
      value: 62.743
    - type: ndcg_at_3
      value: 51.964
    - type: ndcg_at_5
      value: 54.773
    - type: precision_at_1
      value: 44.513999999999996
    - type: precision_at_10
      value: 9.423
    - type: precision_at_100
      value: 1.2309999999999999
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 23.323
    - type: precision_at_5
      value: 16.163
    - type: recall_at_1
      value: 38.83
    - type: recall_at_10
      value: 72.327
    - type: recall_at_100
      value: 89.519
    - type: recall_at_1000
      value: 97.041
    - type: recall_at_3
      value: 57.206
    - type: recall_at_5
      value: 63.88399999999999
    - type: map_at_1
      value: 25.484
    - type: map_at_10
      value: 34.527
    - type: map_at_100
      value: 35.661
    - type: map_at_1000
      value: 35.739
    - type: map_at_3
      value: 32.199
    - type: map_at_5
      value: 33.632
    - type: mrr_at_1
      value: 27.458
    - type: mrr_at_10
      value: 36.543
    - type: mrr_at_100
      value: 37.482
    - type: mrr_at_1000
      value: 37.543
    - type: mrr_at_3
      value: 34.256
    - type: mrr_at_5
      value: 35.618
    - type: ndcg_at_1
      value: 27.458
    - type: ndcg_at_10
      value: 39.396
    - type: ndcg_at_100
      value: 44.742
    - type: ndcg_at_1000
      value: 46.708
    - type: ndcg_at_3
      value: 34.817
    - type: ndcg_at_5
      value: 37.247
    - type: precision_at_1
      value: 27.458
    - type: precision_at_10
      value: 5.976999999999999
    - type: precision_at_100
      value: 0.907
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 14.878
    - type: precision_at_5
      value: 10.35
    - type: recall_at_1
      value: 25.484
    - type: recall_at_10
      value: 52.317
    - type: recall_at_100
      value: 76.701
    - type: recall_at_1000
      value: 91.408
    - type: recall_at_3
      value: 40.043
    - type: recall_at_5
      value: 45.879
    - type: map_at_1
      value: 16.719
    - type: map_at_10
      value: 25.269000000000002
    - type: map_at_100
      value: 26.442
    - type: map_at_1000
      value: 26.557
    - type: map_at_3
      value: 22.56
    - type: map_at_5
      value: 24.082
    - type: mrr_at_1
      value: 20.896
    - type: mrr_at_10
      value: 29.982999999999997
    - type: mrr_at_100
      value: 30.895
    - type: mrr_at_1000
      value: 30.961
    - type: mrr_at_3
      value: 27.239
    - type: mrr_at_5
      value: 28.787000000000003
    - type: ndcg_at_1
      value: 20.896
    - type: ndcg_at_10
      value: 30.814000000000004
    - type: ndcg_at_100
      value: 36.418
    - type: ndcg_at_1000
      value: 39.182
    - type: ndcg_at_3
      value: 25.807999999999996
    - type: ndcg_at_5
      value: 28.143
    - type: precision_at_1
      value: 20.896
    - type: precision_at_10
      value: 5.821
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 12.562000000000001
    - type: precision_at_5
      value: 9.254
    - type: recall_at_1
      value: 16.719
    - type: recall_at_10
      value: 43.155
    - type: recall_at_100
      value: 67.831
    - type: recall_at_1000
      value: 87.617
    - type: recall_at_3
      value: 29.259
    - type: recall_at_5
      value: 35.260999999999996
    - type: map_at_1
      value: 29.398999999999997
    - type: map_at_10
      value: 39.876
    - type: map_at_100
      value: 41.205999999999996
    - type: map_at_1000
      value: 41.321999999999996
    - type: map_at_3
      value: 36.588
    - type: map_at_5
      value: 38.538
    - type: mrr_at_1
      value: 35.9
    - type: mrr_at_10
      value: 45.528
    - type: mrr_at_100
      value: 46.343
    - type: mrr_at_1000
      value: 46.388
    - type: mrr_at_3
      value: 42.862
    - type: mrr_at_5
      value: 44.440000000000005
    - type: ndcg_at_1
      value: 35.9
    - type: ndcg_at_10
      value: 45.987
    - type: ndcg_at_100
      value: 51.370000000000005
    - type: ndcg_at_1000
      value: 53.400000000000006
    - type: ndcg_at_3
      value: 40.841
    - type: ndcg_at_5
      value: 43.447
    - type: precision_at_1
      value: 35.9
    - type: precision_at_10
      value: 8.393
    - type: precision_at_100
      value: 1.283
    - type: precision_at_1000
      value: 0.166
    - type: precision_at_3
      value: 19.538
    - type: precision_at_5
      value: 13.975000000000001
    - type: recall_at_1
      value: 29.398999999999997
    - type: recall_at_10
      value: 58.361
    - type: recall_at_100
      value: 81.081
    - type: recall_at_1000
      value: 94.004
    - type: recall_at_3
      value: 43.657000000000004
    - type: recall_at_5
      value: 50.519999999999996
    - type: map_at_1
      value: 21.589
    - type: map_at_10
      value: 31.608999999999998
    - type: map_at_100
      value: 33.128
    - type: map_at_1000
      value: 33.247
    - type: map_at_3
      value: 28.671999999999997
    - type: map_at_5
      value: 30.233999999999998
    - type: mrr_at_1
      value: 26.712000000000003
    - type: mrr_at_10
      value: 36.713
    - type: mrr_at_100
      value: 37.713
    - type: mrr_at_1000
      value: 37.771
    - type: mrr_at_3
      value: 34.075
    - type: mrr_at_5
      value: 35.451
    - type: ndcg_at_1
      value: 26.712000000000003
    - type: ndcg_at_10
      value: 37.519999999999996
    - type: ndcg_at_100
      value: 43.946000000000005
    - type: ndcg_at_1000
      value: 46.297
    - type: ndcg_at_3
      value: 32.551
    - type: ndcg_at_5
      value: 34.660999999999994
    - type: precision_at_1
      value: 26.712000000000003
    - type: precision_at_10
      value: 7.066
    - type: precision_at_100
      value: 1.216
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 15.906
    - type: precision_at_5
      value: 11.437999999999999
    - type: recall_at_1
      value: 21.589
    - type: recall_at_10
      value: 50.090999999999994
    - type: recall_at_100
      value: 77.43900000000001
    - type: recall_at_1000
      value: 93.35900000000001
    - type: recall_at_3
      value: 36.028999999999996
    - type: recall_at_5
      value: 41.698
    - type: map_at_1
      value: 25.121666666666663
    - type: map_at_10
      value: 34.46258333333334
    - type: map_at_100
      value: 35.710499999999996
    - type: map_at_1000
      value: 35.82691666666666
    - type: map_at_3
      value: 31.563249999999996
    - type: map_at_5
      value: 33.189750000000004
    - type: mrr_at_1
      value: 29.66441666666667
    - type: mrr_at_10
      value: 38.5455
    - type: mrr_at_100
      value: 39.39566666666667
    - type: mrr_at_1000
      value: 39.45325
    - type: mrr_at_3
      value: 36.003333333333345
    - type: mrr_at_5
      value: 37.440916666666666
    - type: ndcg_at_1
      value: 29.66441666666667
    - type: ndcg_at_10
      value: 39.978416666666675
    - type: ndcg_at_100
      value: 45.278666666666666
    - type: ndcg_at_1000
      value: 47.52275
    - type: ndcg_at_3
      value: 35.00058333333334
    - type: ndcg_at_5
      value: 37.34908333333333
    - type: precision_at_1
      value: 29.66441666666667
    - type: precision_at_10
      value: 7.094500000000001
    - type: precision_at_100
      value: 1.1523333333333332
    - type: precision_at_1000
      value: 0.15358333333333332
    - type: precision_at_3
      value: 16.184166666666663
    - type: precision_at_5
      value: 11.6005
    - type: recall_at_1
      value: 25.121666666666663
    - type: recall_at_10
      value: 52.23975000000001
    - type: recall_at_100
      value: 75.48408333333333
    - type: recall_at_1000
      value: 90.95316666666668
    - type: recall_at_3
      value: 38.38458333333333
    - type: recall_at_5
      value: 44.39933333333333
    - type: map_at_1
      value: 23.569000000000003
    - type: map_at_10
      value: 30.389
    - type: map_at_100
      value: 31.396
    - type: map_at_1000
      value: 31.493
    - type: map_at_3
      value: 28.276
    - type: map_at_5
      value: 29.459000000000003
    - type: mrr_at_1
      value: 26.534000000000002
    - type: mrr_at_10
      value: 33.217999999999996
    - type: mrr_at_100
      value: 34.054
    - type: mrr_at_1000
      value: 34.12
    - type: mrr_at_3
      value: 31.058000000000003
    - type: mrr_at_5
      value: 32.330999999999996
    - type: ndcg_at_1
      value: 26.534000000000002
    - type: ndcg_at_10
      value: 34.608
    - type: ndcg_at_100
      value: 39.391999999999996
    - type: ndcg_at_1000
      value: 41.837999999999994
    - type: ndcg_at_3
      value: 30.564999999999998
    - type: ndcg_at_5
      value: 32.509
    - type: precision_at_1
      value: 26.534000000000002
    - type: precision_at_10
      value: 5.414
    - type: precision_at_100
      value: 0.847
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 12.986
    - type: precision_at_5
      value: 9.202
    - type: recall_at_1
      value: 23.569000000000003
    - type: recall_at_10
      value: 44.896
    - type: recall_at_100
      value: 66.476
    - type: recall_at_1000
      value: 84.548
    - type: recall_at_3
      value: 33.79
    - type: recall_at_5
      value: 38.512
    - type: map_at_1
      value: 16.36
    - type: map_at_10
      value: 23.57
    - type: map_at_100
      value: 24.698999999999998
    - type: map_at_1000
      value: 24.834999999999997
    - type: map_at_3
      value: 21.093
    - type: map_at_5
      value: 22.418
    - type: mrr_at_1
      value: 19.718
    - type: mrr_at_10
      value: 27.139999999999997
    - type: mrr_at_100
      value: 28.097
    - type: mrr_at_1000
      value: 28.177999999999997
    - type: mrr_at_3
      value: 24.805
    - type: mrr_at_5
      value: 26.121
    - type: ndcg_at_1
      value: 19.718
    - type: ndcg_at_10
      value: 28.238999999999997
    - type: ndcg_at_100
      value: 33.663
    - type: ndcg_at_1000
      value: 36.763
    - type: ndcg_at_3
      value: 23.747
    - type: ndcg_at_5
      value: 25.796000000000003
    - type: precision_at_1
      value: 19.718
    - type: precision_at_10
      value: 5.282
    - type: precision_at_100
      value: 0.9390000000000001
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 11.264000000000001
    - type: precision_at_5
      value: 8.341
    - type: recall_at_1
      value: 16.36
    - type: recall_at_10
      value: 38.669
    - type: recall_at_100
      value: 63.184
    - type: recall_at_1000
      value: 85.33800000000001
    - type: recall_at_3
      value: 26.214
    - type: recall_at_5
      value: 31.423000000000002
    - type: map_at_1
      value: 25.618999999999996
    - type: map_at_10
      value: 34.361999999999995
    - type: map_at_100
      value: 35.534
    - type: map_at_1000
      value: 35.634
    - type: map_at_3
      value: 31.402
    - type: map_at_5
      value: 32.815
    - type: mrr_at_1
      value: 30.037000000000003
    - type: mrr_at_10
      value: 38.284
    - type: mrr_at_100
      value: 39.141999999999996
    - type: mrr_at_1000
      value: 39.2
    - type: mrr_at_3
      value: 35.603
    - type: mrr_at_5
      value: 36.867
    - type: ndcg_at_1
      value: 30.037000000000003
    - type: ndcg_at_10
      value: 39.87
    - type: ndcg_at_100
      value: 45.243
    - type: ndcg_at_1000
      value: 47.507
    - type: ndcg_at_3
      value: 34.371
    - type: ndcg_at_5
      value: 36.521
    - type: precision_at_1
      value: 30.037000000000003
    - type: precision_at_10
      value: 6.819
    - type: precision_at_100
      value: 1.0699999999999998
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 15.392
    - type: precision_at_5
      value: 10.821
    - type: recall_at_1
      value: 25.618999999999996
    - type: recall_at_10
      value: 52.869
    - type: recall_at_100
      value: 76.395
    - type: recall_at_1000
      value: 92.19500000000001
    - type: recall_at_3
      value: 37.943
    - type: recall_at_5
      value: 43.342999999999996
    - type: map_at_1
      value: 23.283
    - type: map_at_10
      value: 32.155
    - type: map_at_100
      value: 33.724
    - type: map_at_1000
      value: 33.939
    - type: map_at_3
      value: 29.018
    - type: map_at_5
      value: 30.864000000000004
    - type: mrr_at_1
      value: 28.063
    - type: mrr_at_10
      value: 36.632
    - type: mrr_at_100
      value: 37.606
    - type: mrr_at_1000
      value: 37.671
    - type: mrr_at_3
      value: 33.992
    - type: mrr_at_5
      value: 35.613
    - type: ndcg_at_1
      value: 28.063
    - type: ndcg_at_10
      value: 38.024
    - type: ndcg_at_100
      value: 44.292
    - type: ndcg_at_1000
      value: 46.818
    - type: ndcg_at_3
      value: 32.965
    - type: ndcg_at_5
      value: 35.562
    - type: precision_at_1
      value: 28.063
    - type: precision_at_10
      value: 7.352
    - type: precision_at_100
      value: 1.514
    - type: precision_at_1000
      value: 0.23800000000000002
    - type: precision_at_3
      value: 15.481
    - type: precision_at_5
      value: 11.542
    - type: recall_at_1
      value: 23.283
    - type: recall_at_10
      value: 49.756
    - type: recall_at_100
      value: 78.05
    - type: recall_at_1000
      value: 93.854
    - type: recall_at_3
      value: 35.408
    - type: recall_at_5
      value: 42.187000000000005
    - type: map_at_1
      value: 19.201999999999998
    - type: map_at_10
      value: 26.826
    - type: map_at_100
      value: 27.961000000000002
    - type: map_at_1000
      value: 28.066999999999997
    - type: map_at_3
      value: 24.237000000000002
    - type: map_at_5
      value: 25.811
    - type: mrr_at_1
      value: 20.887
    - type: mrr_at_10
      value: 28.660000000000004
    - type: mrr_at_100
      value: 29.660999999999998
    - type: mrr_at_1000
      value: 29.731
    - type: mrr_at_3
      value: 26.155
    - type: mrr_at_5
      value: 27.68
    - type: ndcg_at_1
      value: 20.887
    - type: ndcg_at_10
      value: 31.523
    - type: ndcg_at_100
      value: 37.055
    - type: ndcg_at_1000
      value: 39.579
    - type: ndcg_at_3
      value: 26.529000000000003
    - type: ndcg_at_5
      value: 29.137
    - type: precision_at_1
      value: 20.887
    - type: precision_at_10
      value: 5.065
    - type: precision_at_100
      value: 0.856
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 11.399
    - type: precision_at_5
      value: 8.392
    - type: recall_at_1
      value: 19.201999999999998
    - type: recall_at_10
      value: 44.285000000000004
    - type: recall_at_100
      value: 69.768
    - type: recall_at_1000
      value: 88.302
    - type: recall_at_3
      value: 30.804
    - type: recall_at_5
      value: 37.039
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
      value: 11.244
    - type: map_at_10
      value: 18.956
    - type: map_at_100
      value: 20.674
    - type: map_at_1000
      value: 20.863
    - type: map_at_3
      value: 15.923000000000002
    - type: map_at_5
      value: 17.518
    - type: mrr_at_1
      value: 25.080999999999996
    - type: mrr_at_10
      value: 35.94
    - type: mrr_at_100
      value: 36.969
    - type: mrr_at_1000
      value: 37.013
    - type: mrr_at_3
      value: 32.617000000000004
    - type: mrr_at_5
      value: 34.682
    - type: ndcg_at_1
      value: 25.080999999999996
    - type: ndcg_at_10
      value: 26.539
    - type: ndcg_at_100
      value: 33.601
    - type: ndcg_at_1000
      value: 37.203
    - type: ndcg_at_3
      value: 21.695999999999998
    - type: ndcg_at_5
      value: 23.567
    - type: precision_at_1
      value: 25.080999999999996
    - type: precision_at_10
      value: 8.143
    - type: precision_at_100
      value: 1.5650000000000002
    - type: precision_at_1000
      value: 0.22300000000000003
    - type: precision_at_3
      value: 15.983
    - type: precision_at_5
      value: 12.417
    - type: recall_at_1
      value: 11.244
    - type: recall_at_10
      value: 31.457
    - type: recall_at_100
      value: 55.92
    - type: recall_at_1000
      value: 76.372
    - type: recall_at_3
      value: 19.784
    - type: recall_at_5
      value: 24.857000000000003
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
      value: 8.595
    - type: map_at_10
      value: 18.75
    - type: map_at_100
      value: 26.354
    - type: map_at_1000
      value: 27.912
    - type: map_at_3
      value: 13.794
    - type: map_at_5
      value: 16.021
    - type: mrr_at_1
      value: 65.75
    - type: mrr_at_10
      value: 73.837
    - type: mrr_at_100
      value: 74.22800000000001
    - type: mrr_at_1000
      value: 74.234
    - type: mrr_at_3
      value: 72.5
    - type: mrr_at_5
      value: 73.387
    - type: ndcg_at_1
      value: 52.625
    - type: ndcg_at_10
      value: 39.101
    - type: ndcg_at_100
      value: 43.836000000000006
    - type: ndcg_at_1000
      value: 51.086
    - type: ndcg_at_3
      value: 44.229
    - type: ndcg_at_5
      value: 41.555
    - type: precision_at_1
      value: 65.75
    - type: precision_at_10
      value: 30.45
    - type: precision_at_100
      value: 9.81
    - type: precision_at_1000
      value: 2.045
    - type: precision_at_3
      value: 48.667
    - type: precision_at_5
      value: 40.8
    - type: recall_at_1
      value: 8.595
    - type: recall_at_10
      value: 24.201
    - type: recall_at_100
      value: 50.096
    - type: recall_at_1000
      value: 72.677
    - type: recall_at_3
      value: 15.212
    - type: recall_at_5
      value: 18.745
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
      value: 46.565
    - type: f1
      value: 41.49914329345582
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
      value: 66.60000000000001
    - type: map_at_10
      value: 76.838
    - type: map_at_100
      value: 77.076
    - type: map_at_1000
      value: 77.09
    - type: map_at_3
      value: 75.545
    - type: map_at_5
      value: 76.39
    - type: mrr_at_1
      value: 71.707
    - type: mrr_at_10
      value: 81.514
    - type: mrr_at_100
      value: 81.64099999999999
    - type: mrr_at_1000
      value: 81.645
    - type: mrr_at_3
      value: 80.428
    - type: mrr_at_5
      value: 81.159
    - type: ndcg_at_1
      value: 71.707
    - type: ndcg_at_10
      value: 81.545
    - type: ndcg_at_100
      value: 82.477
    - type: ndcg_at_1000
      value: 82.73899999999999
    - type: ndcg_at_3
      value: 79.292
    - type: ndcg_at_5
      value: 80.599
    - type: precision_at_1
      value: 71.707
    - type: precision_at_10
      value: 10.035
    - type: precision_at_100
      value: 1.068
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 30.918
    - type: precision_at_5
      value: 19.328
    - type: recall_at_1
      value: 66.60000000000001
    - type: recall_at_10
      value: 91.353
    - type: recall_at_100
      value: 95.21
    - type: recall_at_1000
      value: 96.89999999999999
    - type: recall_at_3
      value: 85.188
    - type: recall_at_5
      value: 88.52
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
      value: 19.338
    - type: map_at_10
      value: 31.752000000000002
    - type: map_at_100
      value: 33.516
    - type: map_at_1000
      value: 33.694
    - type: map_at_3
      value: 27.716
    - type: map_at_5
      value: 29.67
    - type: mrr_at_1
      value: 38.117000000000004
    - type: mrr_at_10
      value: 47.323
    - type: mrr_at_100
      value: 48.13
    - type: mrr_at_1000
      value: 48.161
    - type: mrr_at_3
      value: 45.062000000000005
    - type: mrr_at_5
      value: 46.358
    - type: ndcg_at_1
      value: 38.117000000000004
    - type: ndcg_at_10
      value: 39.353
    - type: ndcg_at_100
      value: 46.044000000000004
    - type: ndcg_at_1000
      value: 49.083
    - type: ndcg_at_3
      value: 35.891
    - type: ndcg_at_5
      value: 36.661
    - type: precision_at_1
      value: 38.117000000000004
    - type: precision_at_10
      value: 11.187999999999999
    - type: precision_at_100
      value: 1.802
    - type: precision_at_1000
      value: 0.234
    - type: precision_at_3
      value: 24.126
    - type: precision_at_5
      value: 17.562
    - type: recall_at_1
      value: 19.338
    - type: recall_at_10
      value: 45.735
    - type: recall_at_100
      value: 71.281
    - type: recall_at_1000
      value: 89.537
    - type: recall_at_3
      value: 32.525
    - type: recall_at_5
      value: 37.671
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
      value: 36.995
    - type: map_at_10
      value: 55.032000000000004
    - type: map_at_100
      value: 55.86
    - type: map_at_1000
      value: 55.932
    - type: map_at_3
      value: 52.125
    - type: map_at_5
      value: 53.884
    - type: mrr_at_1
      value: 73.991
    - type: mrr_at_10
      value: 80.096
    - type: mrr_at_100
      value: 80.32000000000001
    - type: mrr_at_1000
      value: 80.331
    - type: mrr_at_3
      value: 79.037
    - type: mrr_at_5
      value: 79.719
    - type: ndcg_at_1
      value: 73.991
    - type: ndcg_at_10
      value: 63.786
    - type: ndcg_at_100
      value: 66.78
    - type: ndcg_at_1000
      value: 68.255
    - type: ndcg_at_3
      value: 59.501000000000005
    - type: ndcg_at_5
      value: 61.82299999999999
    - type: precision_at_1
      value: 73.991
    - type: precision_at_10
      value: 13.157
    - type: precision_at_100
      value: 1.552
    - type: precision_at_1000
      value: 0.17500000000000002
    - type: precision_at_3
      value: 37.519999999999996
    - type: precision_at_5
      value: 24.351
    - type: recall_at_1
      value: 36.995
    - type: recall_at_10
      value: 65.78699999999999
    - type: recall_at_100
      value: 77.583
    - type: recall_at_1000
      value: 87.421
    - type: recall_at_3
      value: 56.279999999999994
    - type: recall_at_5
      value: 60.878
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
      value: 86.80239999999999
    - type: ap
      value: 81.97305141128378
    - type: f1
      value: 86.76976305549273
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
      value: 21.166
    - type: map_at_10
      value: 33.396
    - type: map_at_100
      value: 34.588
    - type: map_at_1000
      value: 34.637
    - type: map_at_3
      value: 29.509999999999998
    - type: map_at_5
      value: 31.719
    - type: mrr_at_1
      value: 21.762
    - type: mrr_at_10
      value: 33.969
    - type: mrr_at_100
      value: 35.099000000000004
    - type: mrr_at_1000
      value: 35.141
    - type: mrr_at_3
      value: 30.148000000000003
    - type: mrr_at_5
      value: 32.324000000000005
    - type: ndcg_at_1
      value: 21.776999999999997
    - type: ndcg_at_10
      value: 40.306999999999995
    - type: ndcg_at_100
      value: 46.068
    - type: ndcg_at_1000
      value: 47.3
    - type: ndcg_at_3
      value: 32.416
    - type: ndcg_at_5
      value: 36.345
    - type: precision_at_1
      value: 21.776999999999997
    - type: precision_at_10
      value: 6.433
    - type: precision_at_100
      value: 0.932
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 13.897
    - type: precision_at_5
      value: 10.324
    - type: recall_at_1
      value: 21.166
    - type: recall_at_10
      value: 61.587
    - type: recall_at_100
      value: 88.251
    - type: recall_at_1000
      value: 97.727
    - type: recall_at_3
      value: 40.196
    - type: recall_at_5
      value: 49.611
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
      value: 93.04605563155496
    - type: f1
      value: 92.78007303978372
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
      value: 69.65116279069767
    - type: f1
      value: 52.75775172527262
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
      value: 70.34633490248822
    - type: f1
      value: 68.15345065392562
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
      value: 75.63887020847343
    - type: f1
      value: 76.08074680233685
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
      value: 33.77933406071333
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
      value: 32.06504927238196
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
      value: 32.20682480490871
    - type: mrr
      value: 33.41462721527003
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
      value: 5.548
    - type: map_at_10
      value: 13.086999999999998
    - type: map_at_100
      value: 16.698
    - type: map_at_1000
      value: 18.151999999999997
    - type: map_at_3
      value: 9.576
    - type: map_at_5
      value: 11.175
    - type: mrr_at_1
      value: 44.272
    - type: mrr_at_10
      value: 53.635999999999996
    - type: mrr_at_100
      value: 54.228
    - type: mrr_at_1000
      value: 54.26499999999999
    - type: mrr_at_3
      value: 51.754
    - type: mrr_at_5
      value: 53.086
    - type: ndcg_at_1
      value: 42.724000000000004
    - type: ndcg_at_10
      value: 34.769
    - type: ndcg_at_100
      value: 32.283
    - type: ndcg_at_1000
      value: 40.843
    - type: ndcg_at_3
      value: 39.852
    - type: ndcg_at_5
      value: 37.858999999999995
    - type: precision_at_1
      value: 44.272
    - type: precision_at_10
      value: 26.068
    - type: precision_at_100
      value: 8.328000000000001
    - type: precision_at_1000
      value: 2.1
    - type: precision_at_3
      value: 37.874
    - type: precision_at_5
      value: 33.065
    - type: recall_at_1
      value: 5.548
    - type: recall_at_10
      value: 16.936999999999998
    - type: recall_at_100
      value: 33.72
    - type: recall_at_1000
      value: 64.348
    - type: recall_at_3
      value: 10.764999999999999
    - type: recall_at_5
      value: 13.361
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
      value: 28.008
    - type: map_at_10
      value: 42.675000000000004
    - type: map_at_100
      value: 43.85
    - type: map_at_1000
      value: 43.884
    - type: map_at_3
      value: 38.286
    - type: map_at_5
      value: 40.78
    - type: mrr_at_1
      value: 31.518
    - type: mrr_at_10
      value: 45.015
    - type: mrr_at_100
      value: 45.924
    - type: mrr_at_1000
      value: 45.946999999999996
    - type: mrr_at_3
      value: 41.348
    - type: mrr_at_5
      value: 43.428
    - type: ndcg_at_1
      value: 31.489
    - type: ndcg_at_10
      value: 50.285999999999994
    - type: ndcg_at_100
      value: 55.291999999999994
    - type: ndcg_at_1000
      value: 56.05
    - type: ndcg_at_3
      value: 41.976
    - type: ndcg_at_5
      value: 46.103
    - type: precision_at_1
      value: 31.489
    - type: precision_at_10
      value: 8.456
    - type: precision_at_100
      value: 1.125
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 19.09
    - type: precision_at_5
      value: 13.841000000000001
    - type: recall_at_1
      value: 28.008
    - type: recall_at_10
      value: 71.21499999999999
    - type: recall_at_100
      value: 92.99
    - type: recall_at_1000
      value: 98.578
    - type: recall_at_3
      value: 49.604
    - type: recall_at_5
      value: 59.094
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
      value: 70.351
    - type: map_at_10
      value: 84.163
    - type: map_at_100
      value: 84.785
    - type: map_at_1000
      value: 84.801
    - type: map_at_3
      value: 81.16
    - type: map_at_5
      value: 83.031
    - type: mrr_at_1
      value: 80.96
    - type: mrr_at_10
      value: 87.241
    - type: mrr_at_100
      value: 87.346
    - type: mrr_at_1000
      value: 87.347
    - type: mrr_at_3
      value: 86.25699999999999
    - type: mrr_at_5
      value: 86.907
    - type: ndcg_at_1
      value: 80.97
    - type: ndcg_at_10
      value: 88.017
    - type: ndcg_at_100
      value: 89.241
    - type: ndcg_at_1000
      value: 89.34299999999999
    - type: ndcg_at_3
      value: 85.053
    - type: ndcg_at_5
      value: 86.663
    - type: precision_at_1
      value: 80.97
    - type: precision_at_10
      value: 13.358
    - type: precision_at_100
      value: 1.525
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.143
    - type: precision_at_5
      value: 24.451999999999998
    - type: recall_at_1
      value: 70.351
    - type: recall_at_10
      value: 95.39800000000001
    - type: recall_at_100
      value: 99.55199999999999
    - type: recall_at_1000
      value: 99.978
    - type: recall_at_3
      value: 86.913
    - type: recall_at_5
      value: 91.448
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
      value: 55.62406719814139
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
      value: 61.386700035141736
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
      value: 4.618
    - type: map_at_10
      value: 12.920000000000002
    - type: map_at_100
      value: 15.304
    - type: map_at_1000
      value: 15.656999999999998
    - type: map_at_3
      value: 9.187
    - type: map_at_5
      value: 10.937
    - type: mrr_at_1
      value: 22.8
    - type: mrr_at_10
      value: 35.13
    - type: mrr_at_100
      value: 36.239
    - type: mrr_at_1000
      value: 36.291000000000004
    - type: mrr_at_3
      value: 31.917
    - type: mrr_at_5
      value: 33.787
    - type: ndcg_at_1
      value: 22.8
    - type: ndcg_at_10
      value: 21.382
    - type: ndcg_at_100
      value: 30.257
    - type: ndcg_at_1000
      value: 36.001
    - type: ndcg_at_3
      value: 20.43
    - type: ndcg_at_5
      value: 17.622
    - type: precision_at_1
      value: 22.8
    - type: precision_at_10
      value: 11.26
    - type: precision_at_100
      value: 2.405
    - type: precision_at_1000
      value: 0.377
    - type: precision_at_3
      value: 19.633
    - type: precision_at_5
      value: 15.68
    - type: recall_at_1
      value: 4.618
    - type: recall_at_10
      value: 22.811999999999998
    - type: recall_at_100
      value: 48.787000000000006
    - type: recall_at_1000
      value: 76.63799999999999
    - type: recall_at_3
      value: 11.952
    - type: recall_at_5
      value: 15.892000000000001
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
      value: 84.01529458252244
    - type: cos_sim_spearman
      value: 77.92985224770254
    - type: euclidean_pearson
      value: 81.04251429422487
    - type: euclidean_spearman
      value: 77.92838490549133
    - type: manhattan_pearson
      value: 80.95892251458979
    - type: manhattan_spearman
      value: 77.81028089705941
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
      value: 83.97885282534388
    - type: cos_sim_spearman
      value: 75.1221970851712
    - type: euclidean_pearson
      value: 80.34455956720097
    - type: euclidean_spearman
      value: 74.5894274239938
    - type: manhattan_pearson
      value: 80.38999766325465
    - type: manhattan_spearman
      value: 74.68524557166975
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
      value: 82.95746064915672
    - type: cos_sim_spearman
      value: 85.08683458043946
    - type: euclidean_pearson
      value: 84.56699492836385
    - type: euclidean_spearman
      value: 85.66089116133713
    - type: manhattan_pearson
      value: 84.47553323458541
    - type: manhattan_spearman
      value: 85.56142206781472
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
      value: 82.71377893595067
    - type: cos_sim_spearman
      value: 81.03453291428589
    - type: euclidean_pearson
      value: 82.57136298308613
    - type: euclidean_spearman
      value: 81.15839961890875
    - type: manhattan_pearson
      value: 82.55157879373837
    - type: manhattan_spearman
      value: 81.1540163767054
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
      value: 86.64197832372373
    - type: cos_sim_spearman
      value: 88.31966852492485
    - type: euclidean_pearson
      value: 87.98692129976983
    - type: euclidean_spearman
      value: 88.6247340837856
    - type: manhattan_pearson
      value: 87.90437827826412
    - type: manhattan_spearman
      value: 88.56278787131457
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
      value: 81.84159950146693
    - type: cos_sim_spearman
      value: 83.90678384140168
    - type: euclidean_pearson
      value: 83.19005018860221
    - type: euclidean_spearman
      value: 84.16260415876295
    - type: manhattan_pearson
      value: 83.05030612994494
    - type: manhattan_spearman
      value: 83.99605629718336
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
      value: 87.49935350176666
    - type: cos_sim_spearman
      value: 87.59086606735383
    - type: euclidean_pearson
      value: 88.06537181129983
    - type: euclidean_spearman
      value: 87.6687448086014
    - type: manhattan_pearson
      value: 87.96599131972935
    - type: manhattan_spearman
      value: 87.63295748969642
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
      value: 67.68232799482763
    - type: cos_sim_spearman
      value: 67.99930378085793
    - type: euclidean_pearson
      value: 68.50275360001696
    - type: euclidean_spearman
      value: 67.81588179309259
    - type: manhattan_pearson
      value: 68.5892154749763
    - type: manhattan_spearman
      value: 67.84357259640682
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
      value: 84.37049618406554
    - type: cos_sim_spearman
      value: 85.57014313159492
    - type: euclidean_pearson
      value: 85.57469513908282
    - type: euclidean_spearman
      value: 85.661948135258
    - type: manhattan_pearson
      value: 85.36866831229028
    - type: manhattan_spearman
      value: 85.5043455368843
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
      value: 84.83259065376154
    - type: mrr
      value: 95.58455433455433
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
      value: 58.817
    - type: map_at_10
      value: 68.459
    - type: map_at_100
      value: 68.951
    - type: map_at_1000
      value: 68.979
    - type: map_at_3
      value: 65.791
    - type: map_at_5
      value: 67.583
    - type: mrr_at_1
      value: 61.667
    - type: mrr_at_10
      value: 69.368
    - type: mrr_at_100
      value: 69.721
    - type: mrr_at_1000
      value: 69.744
    - type: mrr_at_3
      value: 67.278
    - type: mrr_at_5
      value: 68.611
    - type: ndcg_at_1
      value: 61.667
    - type: ndcg_at_10
      value: 72.70100000000001
    - type: ndcg_at_100
      value: 74.928
    - type: ndcg_at_1000
      value: 75.553
    - type: ndcg_at_3
      value: 68.203
    - type: ndcg_at_5
      value: 70.804
    - type: precision_at_1
      value: 61.667
    - type: precision_at_10
      value: 9.533
    - type: precision_at_100
      value: 1.077
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 26.444000000000003
    - type: precision_at_5
      value: 17.599999999999998
    - type: recall_at_1
      value: 58.817
    - type: recall_at_10
      value: 84.789
    - type: recall_at_100
      value: 95.0
    - type: recall_at_1000
      value: 99.667
    - type: recall_at_3
      value: 72.8
    - type: recall_at_5
      value: 79.294
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
      value: 99.8108910891089
    - type: cos_sim_ap
      value: 95.5743678558349
    - type: cos_sim_f1
      value: 90.43133366385722
    - type: cos_sim_precision
      value: 89.67551622418878
    - type: cos_sim_recall
      value: 91.2
    - type: dot_accuracy
      value: 99.75841584158415
    - type: dot_ap
      value: 94.00786363627253
    - type: dot_f1
      value: 87.51910341314316
    - type: dot_precision
      value: 89.20041536863967
    - type: dot_recall
      value: 85.9
    - type: euclidean_accuracy
      value: 99.81485148514851
    - type: euclidean_ap
      value: 95.4752113136905
    - type: euclidean_f1
      value: 90.44334975369456
    - type: euclidean_precision
      value: 89.126213592233
    - type: euclidean_recall
      value: 91.8
    - type: manhattan_accuracy
      value: 99.81584158415842
    - type: manhattan_ap
      value: 95.5163172682464
    - type: manhattan_f1
      value: 90.51987767584097
    - type: manhattan_precision
      value: 92.3076923076923
    - type: manhattan_recall
      value: 88.8
    - type: max_accuracy
      value: 99.81584158415842
    - type: max_ap
      value: 95.5743678558349
    - type: max_f1
      value: 90.51987767584097
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
      value: 62.63235986949449
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
      value: 36.334795589585575
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
      value: 52.02955214518782
    - type: mrr
      value: 52.8004838298956
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
      value: 30.63769566275453
    - type: cos_sim_spearman
      value: 30.422379185989335
    - type: dot_pearson
      value: 26.88493071882256
    - type: dot_spearman
      value: 26.505249740971305
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
      value: 0.21
    - type: map_at_10
      value: 1.654
    - type: map_at_100
      value: 10.095
    - type: map_at_1000
      value: 25.808999999999997
    - type: map_at_3
      value: 0.594
    - type: map_at_5
      value: 0.9289999999999999
    - type: mrr_at_1
      value: 78.0
    - type: mrr_at_10
      value: 87.019
    - type: mrr_at_100
      value: 87.019
    - type: mrr_at_1000
      value: 87.019
    - type: mrr_at_3
      value: 86.333
    - type: mrr_at_5
      value: 86.733
    - type: ndcg_at_1
      value: 73.0
    - type: ndcg_at_10
      value: 66.52900000000001
    - type: ndcg_at_100
      value: 53.433
    - type: ndcg_at_1000
      value: 51.324000000000005
    - type: ndcg_at_3
      value: 72.02199999999999
    - type: ndcg_at_5
      value: 69.696
    - type: precision_at_1
      value: 78.0
    - type: precision_at_10
      value: 70.39999999999999
    - type: precision_at_100
      value: 55.46
    - type: precision_at_1000
      value: 22.758
    - type: precision_at_3
      value: 76.667
    - type: precision_at_5
      value: 74.0
    - type: recall_at_1
      value: 0.21
    - type: recall_at_10
      value: 1.8849999999999998
    - type: recall_at_100
      value: 13.801
    - type: recall_at_1000
      value: 49.649
    - type: recall_at_3
      value: 0.632
    - type: recall_at_5
      value: 1.009
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
      value: 1.797
    - type: map_at_10
      value: 9.01
    - type: map_at_100
      value: 14.682
    - type: map_at_1000
      value: 16.336000000000002
    - type: map_at_3
      value: 4.546
    - type: map_at_5
      value: 5.9270000000000005
    - type: mrr_at_1
      value: 24.490000000000002
    - type: mrr_at_10
      value: 41.156
    - type: mrr_at_100
      value: 42.392
    - type: mrr_at_1000
      value: 42.408
    - type: mrr_at_3
      value: 38.775999999999996
    - type: mrr_at_5
      value: 40.102
    - type: ndcg_at_1
      value: 21.429000000000002
    - type: ndcg_at_10
      value: 22.222
    - type: ndcg_at_100
      value: 34.405
    - type: ndcg_at_1000
      value: 46.599000000000004
    - type: ndcg_at_3
      value: 25.261
    - type: ndcg_at_5
      value: 22.695999999999998
    - type: precision_at_1
      value: 24.490000000000002
    - type: precision_at_10
      value: 19.796
    - type: precision_at_100
      value: 7.306
    - type: precision_at_1000
      value: 1.5350000000000001
    - type: precision_at_3
      value: 27.211000000000002
    - type: precision_at_5
      value: 22.857
    - type: recall_at_1
      value: 1.797
    - type: recall_at_10
      value: 15.706000000000001
    - type: recall_at_100
      value: 46.412
    - type: recall_at_1000
      value: 83.159
    - type: recall_at_3
      value: 6.1370000000000005
    - type: recall_at_5
      value: 8.599
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
      value: 70.3302
    - type: ap
      value: 14.169121204575601
    - type: f1
      value: 54.229345975274235
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
      value: 58.22297679683077
    - type: f1
      value: 58.62984908377875
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
      value: 49.952922428464255
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
      value: 84.68140907194373
    - type: cos_sim_ap
      value: 70.12180123666836
    - type: cos_sim_f1
      value: 65.77501791258658
    - type: cos_sim_precision
      value: 60.07853403141361
    - type: cos_sim_recall
      value: 72.66490765171504
    - type: dot_accuracy
      value: 81.92167848840674
    - type: dot_ap
      value: 60.49837581423469
    - type: dot_f1
      value: 58.44186046511628
    - type: dot_precision
      value: 52.24532224532224
    - type: dot_recall
      value: 66.3060686015831
    - type: euclidean_accuracy
      value: 84.73505394289802
    - type: euclidean_ap
      value: 70.3278904593286
    - type: euclidean_f1
      value: 65.98851124940161
    - type: euclidean_precision
      value: 60.38107752956636
    - type: euclidean_recall
      value: 72.74406332453826
    - type: manhattan_accuracy
      value: 84.73505394289802
    - type: manhattan_ap
      value: 70.00737738537337
    - type: manhattan_f1
      value: 65.80150784822642
    - type: manhattan_precision
      value: 61.892583120204606
    - type: manhattan_recall
      value: 70.23746701846966
    - type: max_accuracy
      value: 84.73505394289802
    - type: max_ap
      value: 70.3278904593286
    - type: max_f1
      value: 65.98851124940161
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
      value: 88.44258159661582
    - type: cos_sim_ap
      value: 84.91926704880888
    - type: cos_sim_f1
      value: 77.07651086632926
    - type: cos_sim_precision
      value: 74.5894554883319
    - type: cos_sim_recall
      value: 79.73514012935017
    - type: dot_accuracy
      value: 85.88116583226608
    - type: dot_ap
      value: 78.9753854779923
    - type: dot_f1
      value: 72.17757637979255
    - type: dot_precision
      value: 66.80647486729143
    - type: dot_recall
      value: 78.48783492454572
    - type: euclidean_accuracy
      value: 88.5299025885823
    - type: euclidean_ap
      value: 85.08006075642194
    - type: euclidean_f1
      value: 77.29637336504163
    - type: euclidean_precision
      value: 74.69836253950014
    - type: euclidean_recall
      value: 80.08161379735141
    - type: manhattan_accuracy
      value: 88.55124771995187
    - type: manhattan_ap
      value: 85.00941529932851
    - type: manhattan_f1
      value: 77.33100233100232
    - type: manhattan_precision
      value: 73.37572573956317
    - type: manhattan_recall
      value: 81.73698798891284
    - type: max_accuracy
      value: 88.55124771995187
    - type: max_ap
      value: 85.08006075642194
    - type: max_f1
      value: 77.33100233100232
---

# gte-small

General Text Embeddings (GTE) model. [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)

The GTE models are trained by Alibaba DAMO Academy. They are mainly based on the BERT framework and currently offer three different sizes of models, including [GTE-large](https://huggingface.co/thenlper/gte-large), [GTE-base](https://huggingface.co/thenlper/gte-base), and [GTE-small](https://huggingface.co/thenlper/gte-small). The GTE models are trained on a large-scale corpus of relevance text pairs, covering a wide range of domains and scenarios. This enables the GTE models to be applied to various downstream tasks of text embeddings, including **information retrieval**, **semantic textual similarity**, **text reranking**, etc.

## Metrics

We compared the performance of the GTE models with other popular text embedding models on the MTEB benchmark. For more detailed comparison results, please refer to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).



| Model Name | Model Size (GB) | Dimension | Sequence Length | Average (56) | Clustering (11) | Pair Classification (3) | Reranking (4) | Retrieval (15) | STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**gte-large**](https://huggingface.co/thenlper/gte-large) | 0.67 | 1024 | 512 | **63.13** | 46.84 | 85.00 | 59.13 | 52.22 | 83.35 | 31.66 | 73.33 |
| [**gte-base**](https://huggingface.co/thenlper/gte-base) 	| 0.22 | 768 | 512 | **62.39** | 46.2 | 84.57 | 58.61 | 51.14 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1.34 | 1024| 512 | 62.25 | 44.49 | 86.03 | 56.61 | 50.56 | 82.05 | 30.19 | 75.24 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.44 | 768 | 512 | 61.5 | 43.80 | 85.73 | 55.91 | 50.29 | 81.05 | 30.28 | 73.84 |
| [**gte-small**](https://huggingface.co/thenlper/gte-small) | 0.07 | 384 | 512 | **61.36** | 44.89 | 83.54 | 57.7 | 49.46 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | - | 1536 | 8192 | 60.99 | 45.9 | 84.89 | 56.32 | 49.25 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.13 | 384 | 512 | 59.93 | 39.92 | 84.67 | 54.32 | 49.04 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | 9.73 | 768 | 512 | 59.51 | 43.72 | 85.06 | 56.42 | 42.24 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	| 0.44 | 768 | 514 	| 57.78 | 43.69 | 83.04 | 59.36 | 43.81 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	| 28.27 | 4096 | 2048 | 57.59 | 38.93 | 81.9 | 55.65 | 48.22 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	| 0.13 | 384 | 512 	| 56.53 | 41.81 | 82.41 | 58.44 | 42.69 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	| 0.09 | 384 | 512 	| 56.26 | 42.35 | 82.37 | 58.04 | 41.95 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	| 0.44 | 768 | 512 	| 56.00 | 41.1 	| 82.54 | 53.14 | 41.88 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	| 0.22 | 768 | 512 	| 55.27 | 40.21 | 85.18 | 53.09 | 33.63 | 81.14 | 31.39 | 69.81 |


## Usage

Code example

```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
```

Use with sentence-transformers:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ['That is a happy person', 'That is a very happy person']

model = SentenceTransformer('thenlper/gte-large')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
```

### Limitation

This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.

### Citation

If you find our paper or models helpful, please consider citing them as follows:

```
@misc{li2023general,
      title={Towards General Text Embeddings with Multi-stage Contrastive Learning}, 
      author={Zehan Li and Xin Zhang and Yanzhao Zhang and Dingkun Long and Pengjun Xie and Meishan Zhang},
      year={2023},
      eprint={2308.03281},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
