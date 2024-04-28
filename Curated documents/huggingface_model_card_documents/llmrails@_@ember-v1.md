---
language: en
license: mit
tags:
- mteb
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
model-index:
- name: ember_v1
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
      value: 76.05970149253731
    - type: ap
      value: 38.76045348512767
    - type: f1
      value: 69.8824007294685
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
      value: 91.977
    - type: ap
      value: 88.63507587170176
    - type: f1
      value: 91.9524133311038
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
      value: 47.938
    - type: f1
      value: 47.58273047536129
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
      value: 41.252
    - type: map_at_10
      value: 56.567
    - type: map_at_100
      value: 57.07600000000001
    - type: map_at_1000
      value: 57.08
    - type: map_at_3
      value: 52.394
    - type: map_at_5
      value: 55.055
    - type: mrr_at_1
      value: 42.39
    - type: mrr_at_10
      value: 57.001999999999995
    - type: mrr_at_100
      value: 57.531
    - type: mrr_at_1000
      value: 57.535000000000004
    - type: mrr_at_3
      value: 52.845
    - type: mrr_at_5
      value: 55.47299999999999
    - type: ndcg_at_1
      value: 41.252
    - type: ndcg_at_10
      value: 64.563
    - type: ndcg_at_100
      value: 66.667
    - type: ndcg_at_1000
      value: 66.77
    - type: ndcg_at_3
      value: 56.120000000000005
    - type: ndcg_at_5
      value: 60.889
    - type: precision_at_1
      value: 41.252
    - type: precision_at_10
      value: 8.982999999999999
    - type: precision_at_100
      value: 0.989
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 22.309
    - type: precision_at_5
      value: 15.690000000000001
    - type: recall_at_1
      value: 41.252
    - type: recall_at_10
      value: 89.82900000000001
    - type: recall_at_100
      value: 98.86200000000001
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 66.927
    - type: recall_at_5
      value: 78.45
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
      value: 48.5799968717232
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
      value: 43.142844164856136
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
      value: 64.45997990276463
    - type: mrr
      value: 77.85560392208592
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
      value: 86.38299310075898
    - type: cos_sim_spearman
      value: 85.81038898286454
    - type: euclidean_pearson
      value: 84.28002556389774
    - type: euclidean_spearman
      value: 85.80315990248238
    - type: manhattan_pearson
      value: 83.9755390675032
    - type: manhattan_spearman
      value: 85.30435335611396
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
      value: 87.89935064935065
    - type: f1
      value: 87.87886687103833
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
      value: 38.84335510371379
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
      value: 36.377963093857005
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
      value: 32.557
    - type: map_at_10
      value: 44.501000000000005
    - type: map_at_100
      value: 46.11
    - type: map_at_1000
      value: 46.232
    - type: map_at_3
      value: 40.711000000000006
    - type: map_at_5
      value: 42.937
    - type: mrr_at_1
      value: 40.916000000000004
    - type: mrr_at_10
      value: 51.317
    - type: mrr_at_100
      value: 52.003
    - type: mrr_at_1000
      value: 52.044999999999995
    - type: mrr_at_3
      value: 48.569
    - type: mrr_at_5
      value: 50.322
    - type: ndcg_at_1
      value: 40.916000000000004
    - type: ndcg_at_10
      value: 51.353
    - type: ndcg_at_100
      value: 56.762
    - type: ndcg_at_1000
      value: 58.555
    - type: ndcg_at_3
      value: 46.064
    - type: ndcg_at_5
      value: 48.677
    - type: precision_at_1
      value: 40.916000000000004
    - type: precision_at_10
      value: 9.927999999999999
    - type: precision_at_100
      value: 1.592
    - type: precision_at_1000
      value: 0.20600000000000002
    - type: precision_at_3
      value: 22.078999999999997
    - type: precision_at_5
      value: 16.08
    - type: recall_at_1
      value: 32.557
    - type: recall_at_10
      value: 63.942
    - type: recall_at_100
      value: 86.436
    - type: recall_at_1000
      value: 97.547
    - type: recall_at_3
      value: 48.367
    - type: recall_at_5
      value: 55.818
    - type: map_at_1
      value: 32.106
    - type: map_at_10
      value: 42.55
    - type: map_at_100
      value: 43.818
    - type: map_at_1000
      value: 43.952999999999996
    - type: map_at_3
      value: 39.421
    - type: map_at_5
      value: 41.276
    - type: mrr_at_1
      value: 39.936
    - type: mrr_at_10
      value: 48.484
    - type: mrr_at_100
      value: 49.123
    - type: mrr_at_1000
      value: 49.163000000000004
    - type: mrr_at_3
      value: 46.221000000000004
    - type: mrr_at_5
      value: 47.603
    - type: ndcg_at_1
      value: 39.936
    - type: ndcg_at_10
      value: 48.25
    - type: ndcg_at_100
      value: 52.674
    - type: ndcg_at_1000
      value: 54.638
    - type: ndcg_at_3
      value: 44.05
    - type: ndcg_at_5
      value: 46.125
    - type: precision_at_1
      value: 39.936
    - type: precision_at_10
      value: 9.096
    - type: precision_at_100
      value: 1.473
    - type: precision_at_1000
      value: 0.19499999999999998
    - type: precision_at_3
      value: 21.295
    - type: precision_at_5
      value: 15.121
    - type: recall_at_1
      value: 32.106
    - type: recall_at_10
      value: 58.107
    - type: recall_at_100
      value: 76.873
    - type: recall_at_1000
      value: 89.079
    - type: recall_at_3
      value: 45.505
    - type: recall_at_5
      value: 51.479
    - type: map_at_1
      value: 41.513
    - type: map_at_10
      value: 54.571999999999996
    - type: map_at_100
      value: 55.579
    - type: map_at_1000
      value: 55.626
    - type: map_at_3
      value: 51.127
    - type: map_at_5
      value: 53.151
    - type: mrr_at_1
      value: 47.398
    - type: mrr_at_10
      value: 57.82000000000001
    - type: mrr_at_100
      value: 58.457
    - type: mrr_at_1000
      value: 58.479000000000006
    - type: mrr_at_3
      value: 55.32899999999999
    - type: mrr_at_5
      value: 56.89999999999999
    - type: ndcg_at_1
      value: 47.398
    - type: ndcg_at_10
      value: 60.599000000000004
    - type: ndcg_at_100
      value: 64.366
    - type: ndcg_at_1000
      value: 65.333
    - type: ndcg_at_3
      value: 54.98
    - type: ndcg_at_5
      value: 57.874
    - type: precision_at_1
      value: 47.398
    - type: precision_at_10
      value: 9.806
    - type: precision_at_100
      value: 1.2590000000000001
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 24.619
    - type: precision_at_5
      value: 16.878
    - type: recall_at_1
      value: 41.513
    - type: recall_at_10
      value: 74.91799999999999
    - type: recall_at_100
      value: 90.96
    - type: recall_at_1000
      value: 97.923
    - type: recall_at_3
      value: 60.013000000000005
    - type: recall_at_5
      value: 67.245
    - type: map_at_1
      value: 26.319
    - type: map_at_10
      value: 35.766999999999996
    - type: map_at_100
      value: 36.765
    - type: map_at_1000
      value: 36.829
    - type: map_at_3
      value: 32.888
    - type: map_at_5
      value: 34.538999999999994
    - type: mrr_at_1
      value: 28.249000000000002
    - type: mrr_at_10
      value: 37.766
    - type: mrr_at_100
      value: 38.62
    - type: mrr_at_1000
      value: 38.667
    - type: mrr_at_3
      value: 35.009
    - type: mrr_at_5
      value: 36.608000000000004
    - type: ndcg_at_1
      value: 28.249000000000002
    - type: ndcg_at_10
      value: 41.215
    - type: ndcg_at_100
      value: 46.274
    - type: ndcg_at_1000
      value: 48.007
    - type: ndcg_at_3
      value: 35.557
    - type: ndcg_at_5
      value: 38.344
    - type: precision_at_1
      value: 28.249000000000002
    - type: precision_at_10
      value: 6.429
    - type: precision_at_100
      value: 0.9480000000000001
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 15.179
    - type: precision_at_5
      value: 10.734
    - type: recall_at_1
      value: 26.319
    - type: recall_at_10
      value: 56.157999999999994
    - type: recall_at_100
      value: 79.65
    - type: recall_at_1000
      value: 92.73
    - type: recall_at_3
      value: 40.738
    - type: recall_at_5
      value: 47.418
    - type: map_at_1
      value: 18.485
    - type: map_at_10
      value: 27.400999999999996
    - type: map_at_100
      value: 28.665000000000003
    - type: map_at_1000
      value: 28.79
    - type: map_at_3
      value: 24.634
    - type: map_at_5
      value: 26.313
    - type: mrr_at_1
      value: 23.134
    - type: mrr_at_10
      value: 32.332
    - type: mrr_at_100
      value: 33.318
    - type: mrr_at_1000
      value: 33.384
    - type: mrr_at_3
      value: 29.664
    - type: mrr_at_5
      value: 31.262
    - type: ndcg_at_1
      value: 23.134
    - type: ndcg_at_10
      value: 33.016
    - type: ndcg_at_100
      value: 38.763
    - type: ndcg_at_1000
      value: 41.619
    - type: ndcg_at_3
      value: 28.017999999999997
    - type: ndcg_at_5
      value: 30.576999999999998
    - type: precision_at_1
      value: 23.134
    - type: precision_at_10
      value: 6.069999999999999
    - type: precision_at_100
      value: 1.027
    - type: precision_at_1000
      value: 0.14200000000000002
    - type: precision_at_3
      value: 13.599
    - type: precision_at_5
      value: 9.975000000000001
    - type: recall_at_1
      value: 18.485
    - type: recall_at_10
      value: 45.39
    - type: recall_at_100
      value: 69.876
    - type: recall_at_1000
      value: 90.023
    - type: recall_at_3
      value: 31.587
    - type: recall_at_5
      value: 38.164
    - type: map_at_1
      value: 30.676
    - type: map_at_10
      value: 41.785
    - type: map_at_100
      value: 43.169000000000004
    - type: map_at_1000
      value: 43.272
    - type: map_at_3
      value: 38.462
    - type: map_at_5
      value: 40.32
    - type: mrr_at_1
      value: 37.729
    - type: mrr_at_10
      value: 47.433
    - type: mrr_at_100
      value: 48.303000000000004
    - type: mrr_at_1000
      value: 48.337
    - type: mrr_at_3
      value: 45.011
    - type: mrr_at_5
      value: 46.455
    - type: ndcg_at_1
      value: 37.729
    - type: ndcg_at_10
      value: 47.921
    - type: ndcg_at_100
      value: 53.477
    - type: ndcg_at_1000
      value: 55.300000000000004
    - type: ndcg_at_3
      value: 42.695
    - type: ndcg_at_5
      value: 45.175
    - type: precision_at_1
      value: 37.729
    - type: precision_at_10
      value: 8.652999999999999
    - type: precision_at_100
      value: 1.336
    - type: precision_at_1000
      value: 0.168
    - type: precision_at_3
      value: 20.18
    - type: precision_at_5
      value: 14.302000000000001
    - type: recall_at_1
      value: 30.676
    - type: recall_at_10
      value: 60.441
    - type: recall_at_100
      value: 83.37
    - type: recall_at_1000
      value: 95.092
    - type: recall_at_3
      value: 45.964
    - type: recall_at_5
      value: 52.319
    - type: map_at_1
      value: 24.978
    - type: map_at_10
      value: 35.926
    - type: map_at_100
      value: 37.341
    - type: map_at_1000
      value: 37.445
    - type: map_at_3
      value: 32.748
    - type: map_at_5
      value: 34.207
    - type: mrr_at_1
      value: 31.163999999999998
    - type: mrr_at_10
      value: 41.394
    - type: mrr_at_100
      value: 42.321
    - type: mrr_at_1000
      value: 42.368
    - type: mrr_at_3
      value: 38.964999999999996
    - type: mrr_at_5
      value: 40.135
    - type: ndcg_at_1
      value: 31.163999999999998
    - type: ndcg_at_10
      value: 42.191
    - type: ndcg_at_100
      value: 48.083999999999996
    - type: ndcg_at_1000
      value: 50.21
    - type: ndcg_at_3
      value: 36.979
    - type: ndcg_at_5
      value: 38.823
    - type: precision_at_1
      value: 31.163999999999998
    - type: precision_at_10
      value: 7.968
    - type: precision_at_100
      value: 1.2550000000000001
    - type: precision_at_1000
      value: 0.16199999999999998
    - type: precision_at_3
      value: 18.075
    - type: precision_at_5
      value: 12.626000000000001
    - type: recall_at_1
      value: 24.978
    - type: recall_at_10
      value: 55.410000000000004
    - type: recall_at_100
      value: 80.562
    - type: recall_at_1000
      value: 94.77600000000001
    - type: recall_at_3
      value: 40.359
    - type: recall_at_5
      value: 45.577
    - type: map_at_1
      value: 26.812166666666666
    - type: map_at_10
      value: 36.706916666666665
    - type: map_at_100
      value: 37.94016666666666
    - type: map_at_1000
      value: 38.05358333333333
    - type: map_at_3
      value: 33.72408333333334
    - type: map_at_5
      value: 35.36508333333333
    - type: mrr_at_1
      value: 31.91516666666667
    - type: mrr_at_10
      value: 41.09716666666666
    - type: mrr_at_100
      value: 41.931916666666666
    - type: mrr_at_1000
      value: 41.98458333333333
    - type: mrr_at_3
      value: 38.60183333333333
    - type: mrr_at_5
      value: 40.031916666666675
    - type: ndcg_at_1
      value: 31.91516666666667
    - type: ndcg_at_10
      value: 42.38725
    - type: ndcg_at_100
      value: 47.56291666666667
    - type: ndcg_at_1000
      value: 49.716499999999996
    - type: ndcg_at_3
      value: 37.36491666666667
    - type: ndcg_at_5
      value: 39.692166666666665
    - type: precision_at_1
      value: 31.91516666666667
    - type: precision_at_10
      value: 7.476749999999999
    - type: precision_at_100
      value: 1.1869166666666668
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 17.275249999999996
    - type: precision_at_5
      value: 12.25825
    - type: recall_at_1
      value: 26.812166666666666
    - type: recall_at_10
      value: 54.82933333333333
    - type: recall_at_100
      value: 77.36508333333333
    - type: recall_at_1000
      value: 92.13366666666667
    - type: recall_at_3
      value: 40.83508333333334
    - type: recall_at_5
      value: 46.85083333333334
    - type: map_at_1
      value: 25.352999999999998
    - type: map_at_10
      value: 33.025999999999996
    - type: map_at_100
      value: 33.882
    - type: map_at_1000
      value: 33.983999999999995
    - type: map_at_3
      value: 30.995
    - type: map_at_5
      value: 32.113
    - type: mrr_at_1
      value: 28.834
    - type: mrr_at_10
      value: 36.14
    - type: mrr_at_100
      value: 36.815
    - type: mrr_at_1000
      value: 36.893
    - type: mrr_at_3
      value: 34.305
    - type: mrr_at_5
      value: 35.263
    - type: ndcg_at_1
      value: 28.834
    - type: ndcg_at_10
      value: 37.26
    - type: ndcg_at_100
      value: 41.723
    - type: ndcg_at_1000
      value: 44.314
    - type: ndcg_at_3
      value: 33.584
    - type: ndcg_at_5
      value: 35.302
    - type: precision_at_1
      value: 28.834
    - type: precision_at_10
      value: 5.736
    - type: precision_at_100
      value: 0.876
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 14.468
    - type: precision_at_5
      value: 9.847
    - type: recall_at_1
      value: 25.352999999999998
    - type: recall_at_10
      value: 47.155
    - type: recall_at_100
      value: 68.024
    - type: recall_at_1000
      value: 87.26899999999999
    - type: recall_at_3
      value: 37.074
    - type: recall_at_5
      value: 41.352
    - type: map_at_1
      value: 17.845
    - type: map_at_10
      value: 25.556
    - type: map_at_100
      value: 26.787
    - type: map_at_1000
      value: 26.913999999999998
    - type: map_at_3
      value: 23.075000000000003
    - type: map_at_5
      value: 24.308
    - type: mrr_at_1
      value: 21.714
    - type: mrr_at_10
      value: 29.543999999999997
    - type: mrr_at_100
      value: 30.543
    - type: mrr_at_1000
      value: 30.618000000000002
    - type: mrr_at_3
      value: 27.174
    - type: mrr_at_5
      value: 28.409000000000002
    - type: ndcg_at_1
      value: 21.714
    - type: ndcg_at_10
      value: 30.562
    - type: ndcg_at_100
      value: 36.27
    - type: ndcg_at_1000
      value: 39.033
    - type: ndcg_at_3
      value: 26.006
    - type: ndcg_at_5
      value: 27.843
    - type: precision_at_1
      value: 21.714
    - type: precision_at_10
      value: 5.657
    - type: precision_at_100
      value: 1
    - type: precision_at_1000
      value: 0.14100000000000001
    - type: precision_at_3
      value: 12.4
    - type: precision_at_5
      value: 8.863999999999999
    - type: recall_at_1
      value: 17.845
    - type: recall_at_10
      value: 41.72
    - type: recall_at_100
      value: 67.06400000000001
    - type: recall_at_1000
      value: 86.515
    - type: recall_at_3
      value: 28.78
    - type: recall_at_5
      value: 33.629999999999995
    - type: map_at_1
      value: 26.695
    - type: map_at_10
      value: 36.205999999999996
    - type: map_at_100
      value: 37.346000000000004
    - type: map_at_1000
      value: 37.447
    - type: map_at_3
      value: 32.84
    - type: map_at_5
      value: 34.733000000000004
    - type: mrr_at_1
      value: 31.343
    - type: mrr_at_10
      value: 40.335
    - type: mrr_at_100
      value: 41.162
    - type: mrr_at_1000
      value: 41.221000000000004
    - type: mrr_at_3
      value: 37.329
    - type: mrr_at_5
      value: 39.068999999999996
    - type: ndcg_at_1
      value: 31.343
    - type: ndcg_at_10
      value: 41.996
    - type: ndcg_at_100
      value: 47.096
    - type: ndcg_at_1000
      value: 49.4
    - type: ndcg_at_3
      value: 35.902
    - type: ndcg_at_5
      value: 38.848
    - type: precision_at_1
      value: 31.343
    - type: precision_at_10
      value: 7.146
    - type: precision_at_100
      value: 1.098
    - type: precision_at_1000
      value: 0.14100000000000001
    - type: precision_at_3
      value: 16.014
    - type: precision_at_5
      value: 11.735
    - type: recall_at_1
      value: 26.695
    - type: recall_at_10
      value: 55.525000000000006
    - type: recall_at_100
      value: 77.376
    - type: recall_at_1000
      value: 93.476
    - type: recall_at_3
      value: 39.439
    - type: recall_at_5
      value: 46.501
    - type: map_at_1
      value: 24.196
    - type: map_at_10
      value: 33.516
    - type: map_at_100
      value: 35.202
    - type: map_at_1000
      value: 35.426
    - type: map_at_3
      value: 30.561
    - type: map_at_5
      value: 31.961000000000002
    - type: mrr_at_1
      value: 29.644
    - type: mrr_at_10
      value: 38.769
    - type: mrr_at_100
      value: 39.843
    - type: mrr_at_1000
      value: 39.888
    - type: mrr_at_3
      value: 36.132999999999996
    - type: mrr_at_5
      value: 37.467
    - type: ndcg_at_1
      value: 29.644
    - type: ndcg_at_10
      value: 39.584
    - type: ndcg_at_100
      value: 45.964
    - type: ndcg_at_1000
      value: 48.27
    - type: ndcg_at_3
      value: 34.577999999999996
    - type: ndcg_at_5
      value: 36.498000000000005
    - type: precision_at_1
      value: 29.644
    - type: precision_at_10
      value: 7.668
    - type: precision_at_100
      value: 1.545
    - type: precision_at_1000
      value: 0.242
    - type: precision_at_3
      value: 16.271
    - type: precision_at_5
      value: 11.620999999999999
    - type: recall_at_1
      value: 24.196
    - type: recall_at_10
      value: 51.171
    - type: recall_at_100
      value: 79.212
    - type: recall_at_1000
      value: 92.976
    - type: recall_at_3
      value: 36.797999999999995
    - type: recall_at_5
      value: 42.006
    - type: map_at_1
      value: 21.023
    - type: map_at_10
      value: 29.677
    - type: map_at_100
      value: 30.618000000000002
    - type: map_at_1000
      value: 30.725
    - type: map_at_3
      value: 27.227
    - type: map_at_5
      value: 28.523
    - type: mrr_at_1
      value: 22.921
    - type: mrr_at_10
      value: 31.832
    - type: mrr_at_100
      value: 32.675
    - type: mrr_at_1000
      value: 32.751999999999995
    - type: mrr_at_3
      value: 29.513
    - type: mrr_at_5
      value: 30.89
    - type: ndcg_at_1
      value: 22.921
    - type: ndcg_at_10
      value: 34.699999999999996
    - type: ndcg_at_100
      value: 39.302
    - type: ndcg_at_1000
      value: 41.919000000000004
    - type: ndcg_at_3
      value: 29.965999999999998
    - type: ndcg_at_5
      value: 32.22
    - type: precision_at_1
      value: 22.921
    - type: precision_at_10
      value: 5.564
    - type: precision_at_100
      value: 0.8340000000000001
    - type: precision_at_1000
      value: 0.11800000000000001
    - type: precision_at_3
      value: 13.123999999999999
    - type: precision_at_5
      value: 9.316
    - type: recall_at_1
      value: 21.023
    - type: recall_at_10
      value: 48.015
    - type: recall_at_100
      value: 68.978
    - type: recall_at_1000
      value: 88.198
    - type: recall_at_3
      value: 35.397
    - type: recall_at_5
      value: 40.701
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
      value: 11.198
    - type: map_at_10
      value: 19.336000000000002
    - type: map_at_100
      value: 21.382
    - type: map_at_1000
      value: 21.581
    - type: map_at_3
      value: 15.992
    - type: map_at_5
      value: 17.613
    - type: mrr_at_1
      value: 25.080999999999996
    - type: mrr_at_10
      value: 36.032
    - type: mrr_at_100
      value: 37.1
    - type: mrr_at_1000
      value: 37.145
    - type: mrr_at_3
      value: 32.595
    - type: mrr_at_5
      value: 34.553
    - type: ndcg_at_1
      value: 25.080999999999996
    - type: ndcg_at_10
      value: 27.290999999999997
    - type: ndcg_at_100
      value: 35.31
    - type: ndcg_at_1000
      value: 38.885
    - type: ndcg_at_3
      value: 21.895999999999997
    - type: ndcg_at_5
      value: 23.669999999999998
    - type: precision_at_1
      value: 25.080999999999996
    - type: precision_at_10
      value: 8.645
    - type: precision_at_100
      value: 1.7209999999999999
    - type: precision_at_1000
      value: 0.23900000000000002
    - type: precision_at_3
      value: 16.287
    - type: precision_at_5
      value: 12.625
    - type: recall_at_1
      value: 11.198
    - type: recall_at_10
      value: 33.355000000000004
    - type: recall_at_100
      value: 60.912
    - type: recall_at_1000
      value: 80.89
    - type: recall_at_3
      value: 20.055
    - type: recall_at_5
      value: 25.14
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
      value: 9.228
    - type: map_at_10
      value: 20.018
    - type: map_at_100
      value: 28.388999999999996
    - type: map_at_1000
      value: 30.073
    - type: map_at_3
      value: 14.366999999999999
    - type: map_at_5
      value: 16.705000000000002
    - type: mrr_at_1
      value: 69
    - type: mrr_at_10
      value: 77.058
    - type: mrr_at_100
      value: 77.374
    - type: mrr_at_1000
      value: 77.384
    - type: mrr_at_3
      value: 75.708
    - type: mrr_at_5
      value: 76.608
    - type: ndcg_at_1
      value: 57.49999999999999
    - type: ndcg_at_10
      value: 41.792
    - type: ndcg_at_100
      value: 47.374
    - type: ndcg_at_1000
      value: 55.13
    - type: ndcg_at_3
      value: 46.353
    - type: ndcg_at_5
      value: 43.702000000000005
    - type: precision_at_1
      value: 69
    - type: precision_at_10
      value: 32.85
    - type: precision_at_100
      value: 10.708
    - type: precision_at_1000
      value: 2.024
    - type: precision_at_3
      value: 49.5
    - type: precision_at_5
      value: 42.05
    - type: recall_at_1
      value: 9.228
    - type: recall_at_10
      value: 25.635
    - type: recall_at_100
      value: 54.894
    - type: recall_at_1000
      value: 79.38
    - type: recall_at_3
      value: 15.68
    - type: recall_at_5
      value: 19.142
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
      value: 52.035
    - type: f1
      value: 46.85325505614071
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
      value: 70.132
    - type: map_at_10
      value: 79.527
    - type: map_at_100
      value: 79.81200000000001
    - type: map_at_1000
      value: 79.828
    - type: map_at_3
      value: 78.191
    - type: map_at_5
      value: 79.092
    - type: mrr_at_1
      value: 75.563
    - type: mrr_at_10
      value: 83.80199999999999
    - type: mrr_at_100
      value: 83.93
    - type: mrr_at_1000
      value: 83.933
    - type: mrr_at_3
      value: 82.818
    - type: mrr_at_5
      value: 83.505
    - type: ndcg_at_1
      value: 75.563
    - type: ndcg_at_10
      value: 83.692
    - type: ndcg_at_100
      value: 84.706
    - type: ndcg_at_1000
      value: 85.001
    - type: ndcg_at_3
      value: 81.51
    - type: ndcg_at_5
      value: 82.832
    - type: precision_at_1
      value: 75.563
    - type: precision_at_10
      value: 10.245
    - type: precision_at_100
      value: 1.0959999999999999
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 31.518
    - type: precision_at_5
      value: 19.772000000000002
    - type: recall_at_1
      value: 70.132
    - type: recall_at_10
      value: 92.204
    - type: recall_at_100
      value: 96.261
    - type: recall_at_1000
      value: 98.17399999999999
    - type: recall_at_3
      value: 86.288
    - type: recall_at_5
      value: 89.63799999999999
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
      value: 22.269
    - type: map_at_10
      value: 36.042
    - type: map_at_100
      value: 37.988
    - type: map_at_1000
      value: 38.162
    - type: map_at_3
      value: 31.691000000000003
    - type: map_at_5
      value: 33.988
    - type: mrr_at_1
      value: 44.907000000000004
    - type: mrr_at_10
      value: 53.348
    - type: mrr_at_100
      value: 54.033
    - type: mrr_at_1000
      value: 54.064
    - type: mrr_at_3
      value: 50.977
    - type: mrr_at_5
      value: 52.112
    - type: ndcg_at_1
      value: 44.907000000000004
    - type: ndcg_at_10
      value: 44.302
    - type: ndcg_at_100
      value: 51.054
    - type: ndcg_at_1000
      value: 53.822
    - type: ndcg_at_3
      value: 40.615
    - type: ndcg_at_5
      value: 41.455999999999996
    - type: precision_at_1
      value: 44.907000000000004
    - type: precision_at_10
      value: 12.176
    - type: precision_at_100
      value: 1.931
    - type: precision_at_1000
      value: 0.243
    - type: precision_at_3
      value: 27.16
    - type: precision_at_5
      value: 19.567999999999998
    - type: recall_at_1
      value: 22.269
    - type: recall_at_10
      value: 51.188
    - type: recall_at_100
      value: 75.924
    - type: recall_at_1000
      value: 92.525
    - type: recall_at_3
      value: 36.643
    - type: recall_at_5
      value: 42.27
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
      value: 40.412
    - type: map_at_10
      value: 66.376
    - type: map_at_100
      value: 67.217
    - type: map_at_1000
      value: 67.271
    - type: map_at_3
      value: 62.741
    - type: map_at_5
      value: 65.069
    - type: mrr_at_1
      value: 80.824
    - type: mrr_at_10
      value: 86.53
    - type: mrr_at_100
      value: 86.67399999999999
    - type: mrr_at_1000
      value: 86.678
    - type: mrr_at_3
      value: 85.676
    - type: mrr_at_5
      value: 86.256
    - type: ndcg_at_1
      value: 80.824
    - type: ndcg_at_10
      value: 74.332
    - type: ndcg_at_100
      value: 77.154
    - type: ndcg_at_1000
      value: 78.12400000000001
    - type: ndcg_at_3
      value: 69.353
    - type: ndcg_at_5
      value: 72.234
    - type: precision_at_1
      value: 80.824
    - type: precision_at_10
      value: 15.652
    - type: precision_at_100
      value: 1.7840000000000003
    - type: precision_at_1000
      value: 0.191
    - type: precision_at_3
      value: 44.911
    - type: precision_at_5
      value: 29.221000000000004
    - type: recall_at_1
      value: 40.412
    - type: recall_at_10
      value: 78.25800000000001
    - type: recall_at_100
      value: 89.196
    - type: recall_at_1000
      value: 95.544
    - type: recall_at_3
      value: 67.367
    - type: recall_at_5
      value: 73.05199999999999
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
      value: 92.78880000000001
    - type: ap
      value: 89.39251741048801
    - type: f1
      value: 92.78019950076781
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
      value: 22.888
    - type: map_at_10
      value: 35.146
    - type: map_at_100
      value: 36.325
    - type: map_at_1000
      value: 36.372
    - type: map_at_3
      value: 31.3
    - type: map_at_5
      value: 33.533
    - type: mrr_at_1
      value: 23.480999999999998
    - type: mrr_at_10
      value: 35.777
    - type: mrr_at_100
      value: 36.887
    - type: mrr_at_1000
      value: 36.928
    - type: mrr_at_3
      value: 31.989
    - type: mrr_at_5
      value: 34.202
    - type: ndcg_at_1
      value: 23.496
    - type: ndcg_at_10
      value: 42.028999999999996
    - type: ndcg_at_100
      value: 47.629
    - type: ndcg_at_1000
      value: 48.785000000000004
    - type: ndcg_at_3
      value: 34.227000000000004
    - type: ndcg_at_5
      value: 38.207
    - type: precision_at_1
      value: 23.496
    - type: precision_at_10
      value: 6.596
    - type: precision_at_100
      value: 0.9400000000000001
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.513000000000002
    - type: precision_at_5
      value: 10.711
    - type: recall_at_1
      value: 22.888
    - type: recall_at_10
      value: 63.129999999999995
    - type: recall_at_100
      value: 88.90299999999999
    - type: recall_at_1000
      value: 97.69
    - type: recall_at_3
      value: 42.014
    - type: recall_at_5
      value: 51.554
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
      value: 94.59188326493388
    - type: f1
      value: 94.36568950290486
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
      value: 79.25672594619242
    - type: f1
      value: 59.52405059722216
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
      value: 77.4142568930733
    - type: f1
      value: 75.23044196543388
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
      value: 80.44720914593141
    - type: f1
      value: 80.41049641537015
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
      value: 31.960921474993775
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
      value: 30.88042240204361
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
      value: 32.27071371606404
    - type: mrr
      value: 33.541450459533856
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
      value: 6.551
    - type: map_at_10
      value: 14.359
    - type: map_at_100
      value: 18.157
    - type: map_at_1000
      value: 19.659
    - type: map_at_3
      value: 10.613999999999999
    - type: map_at_5
      value: 12.296
    - type: mrr_at_1
      value: 47.368
    - type: mrr_at_10
      value: 56.689
    - type: mrr_at_100
      value: 57.24399999999999
    - type: mrr_at_1000
      value: 57.284
    - type: mrr_at_3
      value: 54.489
    - type: mrr_at_5
      value: 55.928999999999995
    - type: ndcg_at_1
      value: 45.511
    - type: ndcg_at_10
      value: 36.911
    - type: ndcg_at_100
      value: 34.241
    - type: ndcg_at_1000
      value: 43.064
    - type: ndcg_at_3
      value: 42.348
    - type: ndcg_at_5
      value: 39.884
    - type: precision_at_1
      value: 46.749
    - type: precision_at_10
      value: 27.028000000000002
    - type: precision_at_100
      value: 8.52
    - type: precision_at_1000
      value: 2.154
    - type: precision_at_3
      value: 39.525
    - type: precision_at_5
      value: 34.18
    - type: recall_at_1
      value: 6.551
    - type: recall_at_10
      value: 18.602
    - type: recall_at_100
      value: 34.882999999999996
    - type: recall_at_1000
      value: 66.049
    - type: recall_at_3
      value: 11.872
    - type: recall_at_5
      value: 14.74
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
      value: 27.828999999999997
    - type: map_at_10
      value: 43.606
    - type: map_at_100
      value: 44.656
    - type: map_at_1000
      value: 44.690000000000005
    - type: map_at_3
      value: 39.015
    - type: map_at_5
      value: 41.625
    - type: mrr_at_1
      value: 31.518
    - type: mrr_at_10
      value: 46.047
    - type: mrr_at_100
      value: 46.846
    - type: mrr_at_1000
      value: 46.867999999999995
    - type: mrr_at_3
      value: 42.154
    - type: mrr_at_5
      value: 44.468999999999994
    - type: ndcg_at_1
      value: 31.518
    - type: ndcg_at_10
      value: 51.768
    - type: ndcg_at_100
      value: 56.184999999999995
    - type: ndcg_at_1000
      value: 56.92
    - type: ndcg_at_3
      value: 43.059999999999995
    - type: ndcg_at_5
      value: 47.481
    - type: precision_at_1
      value: 31.518
    - type: precision_at_10
      value: 8.824
    - type: precision_at_100
      value: 1.131
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 19.969
    - type: precision_at_5
      value: 14.502
    - type: recall_at_1
      value: 27.828999999999997
    - type: recall_at_10
      value: 74.244
    - type: recall_at_100
      value: 93.325
    - type: recall_at_1000
      value: 98.71799999999999
    - type: recall_at_3
      value: 51.601
    - type: recall_at_5
      value: 61.841
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
      value: 71.54
    - type: map_at_10
      value: 85.509
    - type: map_at_100
      value: 86.137
    - type: map_at_1000
      value: 86.151
    - type: map_at_3
      value: 82.624
    - type: map_at_5
      value: 84.425
    - type: mrr_at_1
      value: 82.45
    - type: mrr_at_10
      value: 88.344
    - type: mrr_at_100
      value: 88.437
    - type: mrr_at_1000
      value: 88.437
    - type: mrr_at_3
      value: 87.417
    - type: mrr_at_5
      value: 88.066
    - type: ndcg_at_1
      value: 82.45
    - type: ndcg_at_10
      value: 89.092
    - type: ndcg_at_100
      value: 90.252
    - type: ndcg_at_1000
      value: 90.321
    - type: ndcg_at_3
      value: 86.404
    - type: ndcg_at_5
      value: 87.883
    - type: precision_at_1
      value: 82.45
    - type: precision_at_10
      value: 13.496
    - type: precision_at_100
      value: 1.536
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.833
    - type: precision_at_5
      value: 24.79
    - type: recall_at_1
      value: 71.54
    - type: recall_at_10
      value: 95.846
    - type: recall_at_100
      value: 99.715
    - type: recall_at_1000
      value: 99.979
    - type: recall_at_3
      value: 88.01299999999999
    - type: recall_at_5
      value: 92.32000000000001
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
      value: 57.60557586253866
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
      value: 64.0287172242051
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
      value: 3.9849999999999994
    - type: map_at_10
      value: 11.397
    - type: map_at_100
      value: 13.985
    - type: map_at_1000
      value: 14.391000000000002
    - type: map_at_3
      value: 7.66
    - type: map_at_5
      value: 9.46
    - type: mrr_at_1
      value: 19.8
    - type: mrr_at_10
      value: 31.958
    - type: mrr_at_100
      value: 33.373999999999995
    - type: mrr_at_1000
      value: 33.411
    - type: mrr_at_3
      value: 28.316999999999997
    - type: mrr_at_5
      value: 30.297
    - type: ndcg_at_1
      value: 19.8
    - type: ndcg_at_10
      value: 19.580000000000002
    - type: ndcg_at_100
      value: 29.555999999999997
    - type: ndcg_at_1000
      value: 35.882
    - type: ndcg_at_3
      value: 17.544
    - type: ndcg_at_5
      value: 15.815999999999999
    - type: precision_at_1
      value: 19.8
    - type: precision_at_10
      value: 10.61
    - type: precision_at_100
      value: 2.501
    - type: precision_at_1000
      value: 0.40099999999999997
    - type: precision_at_3
      value: 16.900000000000002
    - type: precision_at_5
      value: 14.44
    - type: recall_at_1
      value: 3.9849999999999994
    - type: recall_at_10
      value: 21.497
    - type: recall_at_100
      value: 50.727999999999994
    - type: recall_at_1000
      value: 81.27499999999999
    - type: recall_at_3
      value: 10.263
    - type: recall_at_5
      value: 14.643
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
      value: 85.0087509585503
    - type: cos_sim_spearman
      value: 81.74697270664319
    - type: euclidean_pearson
      value: 81.80424382731947
    - type: euclidean_spearman
      value: 81.29794251968431
    - type: manhattan_pearson
      value: 81.81524666226125
    - type: manhattan_spearman
      value: 81.29475370198963
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
      value: 86.44442736429552
    - type: cos_sim_spearman
      value: 78.51011398910948
    - type: euclidean_pearson
      value: 83.36181801196723
    - type: euclidean_spearman
      value: 79.47272621331535
    - type: manhattan_pearson
      value: 83.3660113483837
    - type: manhattan_spearman
      value: 79.47695922566032
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
      value: 85.82923943323635
    - type: cos_sim_spearman
      value: 86.62037823380983
    - type: euclidean_pearson
      value: 83.56369548403958
    - type: euclidean_spearman
      value: 84.2176755481191
    - type: manhattan_pearson
      value: 83.55460702084464
    - type: manhattan_spearman
      value: 84.18617930921467
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
      value: 84.09071068110103
    - type: cos_sim_spearman
      value: 83.05697553913335
    - type: euclidean_pearson
      value: 81.1377457216497
    - type: euclidean_spearman
      value: 81.74714169016676
    - type: manhattan_pearson
      value: 81.0893424142723
    - type: manhattan_spearman
      value: 81.7058918219677
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
      value: 87.61132157220429
    - type: cos_sim_spearman
      value: 88.38581627185445
    - type: euclidean_pearson
      value: 86.14904510913374
    - type: euclidean_spearman
      value: 86.5452758925542
    - type: manhattan_pearson
      value: 86.1484025377679
    - type: manhattan_spearman
      value: 86.55483841566252
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
      value: 85.46195145161064
    - type: cos_sim_spearman
      value: 86.82409112251158
    - type: euclidean_pearson
      value: 84.75479672288957
    - type: euclidean_spearman
      value: 85.41144307151548
    - type: manhattan_pearson
      value: 84.70914329694165
    - type: manhattan_spearman
      value: 85.38477943384089
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
      value: 88.06351289930238
    - type: cos_sim_spearman
      value: 87.90311138579116
    - type: euclidean_pearson
      value: 86.17651467063077
    - type: euclidean_spearman
      value: 84.89447802019073
    - type: manhattan_pearson
      value: 86.3267677479595
    - type: manhattan_spearman
      value: 85.00472295103874
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
      value: 67.78311975978767
    - type: cos_sim_spearman
      value: 66.76465685245887
    - type: euclidean_pearson
      value: 67.21687806595443
    - type: euclidean_spearman
      value: 65.05776733534435
    - type: manhattan_pearson
      value: 67.14008143635883
    - type: manhattan_spearman
      value: 65.25247076149701
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
      value: 86.7403488889418
    - type: cos_sim_spearman
      value: 87.76870289783061
    - type: euclidean_pearson
      value: 84.83171077794671
    - type: euclidean_spearman
      value: 85.50579695091902
    - type: manhattan_pearson
      value: 84.83074260180555
    - type: manhattan_spearman
      value: 85.47589026938667
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
      value: 87.56234016237356
    - type: mrr
      value: 96.26124238869338
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
      value: 59.660999999999994
    - type: map_at_10
      value: 69.105
    - type: map_at_100
      value: 69.78
    - type: map_at_1000
      value: 69.80199999999999
    - type: map_at_3
      value: 65.991
    - type: map_at_5
      value: 68.02
    - type: mrr_at_1
      value: 62.666999999999994
    - type: mrr_at_10
      value: 70.259
    - type: mrr_at_100
      value: 70.776
    - type: mrr_at_1000
      value: 70.796
    - type: mrr_at_3
      value: 67.889
    - type: mrr_at_5
      value: 69.52199999999999
    - type: ndcg_at_1
      value: 62.666999999999994
    - type: ndcg_at_10
      value: 73.425
    - type: ndcg_at_100
      value: 75.955
    - type: ndcg_at_1000
      value: 76.459
    - type: ndcg_at_3
      value: 68.345
    - type: ndcg_at_5
      value: 71.319
    - type: precision_at_1
      value: 62.666999999999994
    - type: precision_at_10
      value: 9.667
    - type: precision_at_100
      value: 1.09
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 26.333000000000002
    - type: precision_at_5
      value: 17.732999999999997
    - type: recall_at_1
      value: 59.660999999999994
    - type: recall_at_10
      value: 85.422
    - type: recall_at_100
      value: 96.167
    - type: recall_at_1000
      value: 100
    - type: recall_at_3
      value: 72.044
    - type: recall_at_5
      value: 79.428
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
      value: 99.86435643564356
    - type: cos_sim_ap
      value: 96.83057412333741
    - type: cos_sim_f1
      value: 93.04215337734891
    - type: cos_sim_precision
      value: 94.53044375644994
    - type: cos_sim_recall
      value: 91.60000000000001
    - type: dot_accuracy
      value: 99.7910891089109
    - type: dot_ap
      value: 94.10681982106397
    - type: dot_f1
      value: 89.34881373043918
    - type: dot_precision
      value: 90.21406727828746
    - type: dot_recall
      value: 88.5
    - type: euclidean_accuracy
      value: 99.85544554455446
    - type: euclidean_ap
      value: 96.78545104478602
    - type: euclidean_f1
      value: 92.65143992055613
    - type: euclidean_precision
      value: 92.01183431952663
    - type: euclidean_recall
      value: 93.30000000000001
    - type: manhattan_accuracy
      value: 99.85841584158416
    - type: manhattan_ap
      value: 96.80748903307823
    - type: manhattan_f1
      value: 92.78247884519662
    - type: manhattan_precision
      value: 92.36868186323092
    - type: manhattan_recall
      value: 93.2
    - type: max_accuracy
      value: 99.86435643564356
    - type: max_ap
      value: 96.83057412333741
    - type: max_f1
      value: 93.04215337734891
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
      value: 65.53971025855282
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
      value: 33.97791591490788
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
      value: 55.852215301355066
    - type: mrr
      value: 56.85527809608691
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
      value: 31.21442519856758
    - type: cos_sim_spearman
      value: 30.822536216936825
    - type: dot_pearson
      value: 28.661325528121807
    - type: dot_spearman
      value: 28.1435226478879
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
      value: 0.183
    - type: map_at_10
      value: 1.526
    - type: map_at_100
      value: 7.915
    - type: map_at_1000
      value: 19.009
    - type: map_at_3
      value: 0.541
    - type: map_at_5
      value: 0.8659999999999999
    - type: mrr_at_1
      value: 68
    - type: mrr_at_10
      value: 81.186
    - type: mrr_at_100
      value: 81.186
    - type: mrr_at_1000
      value: 81.186
    - type: mrr_at_3
      value: 80
    - type: mrr_at_5
      value: 80.9
    - type: ndcg_at_1
      value: 64
    - type: ndcg_at_10
      value: 64.13799999999999
    - type: ndcg_at_100
      value: 47.632000000000005
    - type: ndcg_at_1000
      value: 43.037
    - type: ndcg_at_3
      value: 67.542
    - type: ndcg_at_5
      value: 67.496
    - type: precision_at_1
      value: 68
    - type: precision_at_10
      value: 67.80000000000001
    - type: precision_at_100
      value: 48.980000000000004
    - type: precision_at_1000
      value: 19.036
    - type: precision_at_3
      value: 72
    - type: precision_at_5
      value: 71.2
    - type: recall_at_1
      value: 0.183
    - type: recall_at_10
      value: 1.799
    - type: recall_at_100
      value: 11.652999999999999
    - type: recall_at_1000
      value: 40.086
    - type: recall_at_3
      value: 0.5930000000000001
    - type: recall_at_5
      value: 0.983
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
      value: 2.29
    - type: map_at_10
      value: 9.489
    - type: map_at_100
      value: 15.051
    - type: map_at_1000
      value: 16.561999999999998
    - type: map_at_3
      value: 5.137
    - type: map_at_5
      value: 6.7989999999999995
    - type: mrr_at_1
      value: 28.571
    - type: mrr_at_10
      value: 45.699
    - type: mrr_at_100
      value: 46.461000000000006
    - type: mrr_at_1000
      value: 46.461000000000006
    - type: mrr_at_3
      value: 41.837
    - type: mrr_at_5
      value: 43.163000000000004
    - type: ndcg_at_1
      value: 23.469
    - type: ndcg_at_10
      value: 23.544999999999998
    - type: ndcg_at_100
      value: 34.572
    - type: ndcg_at_1000
      value: 46.035
    - type: ndcg_at_3
      value: 27.200000000000003
    - type: ndcg_at_5
      value: 25.266
    - type: precision_at_1
      value: 28.571
    - type: precision_at_10
      value: 22.041
    - type: precision_at_100
      value: 7.3469999999999995
    - type: precision_at_1000
      value: 1.484
    - type: precision_at_3
      value: 29.932
    - type: precision_at_5
      value: 26.531
    - type: recall_at_1
      value: 2.29
    - type: recall_at_10
      value: 15.895999999999999
    - type: recall_at_100
      value: 45.518
    - type: recall_at_1000
      value: 80.731
    - type: recall_at_3
      value: 6.433
    - type: recall_at_5
      value: 9.484
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
      value: 71.4178
    - type: ap
      value: 14.575240629602373
    - type: f1
      value: 55.02449563229096
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
      value: 60.00282965478212
    - type: f1
      value: 60.34413028768773
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
      value: 50.409448342549936
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
      value: 87.62591643321214
    - type: cos_sim_ap
      value: 79.28766491329633
    - type: cos_sim_f1
      value: 71.98772064466617
    - type: cos_sim_precision
      value: 69.8609731876862
    - type: cos_sim_recall
      value: 74.24802110817942
    - type: dot_accuracy
      value: 84.75293556654945
    - type: dot_ap
      value: 69.72705761174353
    - type: dot_f1
      value: 65.08692852543464
    - type: dot_precision
      value: 63.57232704402516
    - type: dot_recall
      value: 66.6754617414248
    - type: euclidean_accuracy
      value: 87.44710019669786
    - type: euclidean_ap
      value: 79.11021477292638
    - type: euclidean_f1
      value: 71.5052389470994
    - type: euclidean_precision
      value: 69.32606541129832
    - type: euclidean_recall
      value: 73.82585751978891
    - type: manhattan_accuracy
      value: 87.42325803182929
    - type: manhattan_ap
      value: 79.05094494327616
    - type: manhattan_f1
      value: 71.36333985649055
    - type: manhattan_precision
      value: 70.58064516129032
    - type: manhattan_recall
      value: 72.16358839050132
    - type: max_accuracy
      value: 87.62591643321214
    - type: max_ap
      value: 79.28766491329633
    - type: max_f1
      value: 71.98772064466617
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
      value: 88.85202002561415
    - type: cos_sim_ap
      value: 85.9835303311168
    - type: cos_sim_f1
      value: 78.25741142443962
    - type: cos_sim_precision
      value: 73.76635768811342
    - type: cos_sim_recall
      value: 83.3307668617185
    - type: dot_accuracy
      value: 88.20584468506229
    - type: dot_ap
      value: 83.591632302697
    - type: dot_f1
      value: 76.81739705396173
    - type: dot_precision
      value: 73.45275728837373
    - type: dot_recall
      value: 80.50508161379734
    - type: euclidean_accuracy
      value: 88.64633057787093
    - type: euclidean_ap
      value: 85.25705123182283
    - type: euclidean_f1
      value: 77.18535726329199
    - type: euclidean_precision
      value: 75.17699437997226
    - type: euclidean_recall
      value: 79.30397289805975
    - type: manhattan_accuracy
      value: 88.63274731245392
    - type: manhattan_ap
      value: 85.2376825633018
    - type: manhattan_f1
      value: 77.15810785937788
    - type: manhattan_precision
      value: 73.92255061014319
    - type: manhattan_recall
      value: 80.68986757006468
    - type: max_accuracy
      value: 88.85202002561415
    - type: max_ap
      value: 85.9835303311168
    - type: max_f1
      value: 78.25741142443962
---

# ember-v1

<p align="center">
<img src="https://console.llmrails.com/assets/img/logo-black.svg" width="150px">
</p>

This model has been trained on an extensive corpus of text pairs that encompass a broad spectrum of domains, including finance, science, medicine, law, and various others. During the training process, we incorporated techniques derived from the [RetroMAE](https://arxiv.org/abs/2205.12035) and [SetFit](https://arxiv.org/abs/2209.11055) research papers.

We are pleased to offer this model as an API service through our platform, [LLMRails](https://llmrails.com/?ref=ember-v1). If you are interested, please don't hesitate to sign up.

### Plans
- The research paper will be published soon.
-  The v2 of the model is currently in development and will feature an extended maximum sequence length of 4,000 tokens.

## Usage
Use with API request:
```bash
curl --location 'https://api.llmrails.com/v1/embeddings' \
--header 'X-API-KEY: {token}' \
--header 'Content-Type: application/json' \
--data '{
   "input": ["This is an example sentence"],
   "model":"embedding-english-v1" # equals to ember-v1
}'
```
API docs: https://docs.llmrails.com/embedding/embed-text<br>
Langchain plugin: https://python.langchain.com/docs/integrations/text_embedding/llm_rails

Use with transformers:
```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = [
    "This is an example sentence",
    "Each sentence is converted"
]

tokenizer = AutoTokenizer.from_pretrained("llmrails/ember-v1")
model = AutoModel.from_pretrained("llmrails/ember-v1")

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

sentences = [
	"This is an example sentence",
    "Each sentence is converted"
]

model = SentenceTransformer('llmrails/ember-v1')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
```

## Massive Text Embedding Benchmark (MTEB) Evaluation
Our model achieve state-of-the-art performance on [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

|                               Model Name                                | Dimension | Sequence Length | Average (56) | 
|:-----------------------------------------------------------------------:|:---------:|:---:|:------------:|
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   1024    |       512       |    64.23     |  
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   768    |       512       |    63.55     |
| [ember-v1](https://huggingface.co/llmrails/emmbedding-en-v1) |   1024    | 512 |    **63.54**     |  
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/types-of-embedding-models) |   1536    |      8191       |    60.99     |

### Limitation

This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.

<img src="https://pixel.llmrails.com/hf/2AtscRthisA1rZzQr8T7Zm">