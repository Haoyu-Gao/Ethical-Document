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
- name: e5-large-v2
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
      value: 79.22388059701493
    - type: ap
      value: 43.20816505595132
    - type: f1
      value: 73.27811303522058
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
      value: 93.748325
    - type: ap
      value: 90.72534979701297
    - type: f1
      value: 93.73895874282185
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
      value: 48.612
    - type: f1
      value: 47.61157345898393
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
      value: 23.541999999999998
    - type: map_at_10
      value: 38.208
    - type: map_at_100
      value: 39.417
    - type: map_at_1000
      value: 39.428999999999995
    - type: map_at_3
      value: 33.95
    - type: map_at_5
      value: 36.329
    - type: mrr_at_1
      value: 23.755000000000003
    - type: mrr_at_10
      value: 38.288
    - type: mrr_at_100
      value: 39.511
    - type: mrr_at_1000
      value: 39.523
    - type: mrr_at_3
      value: 34.009
    - type: mrr_at_5
      value: 36.434
    - type: ndcg_at_1
      value: 23.541999999999998
    - type: ndcg_at_10
      value: 46.417
    - type: ndcg_at_100
      value: 51.812000000000005
    - type: ndcg_at_1000
      value: 52.137
    - type: ndcg_at_3
      value: 37.528
    - type: ndcg_at_5
      value: 41.81
    - type: precision_at_1
      value: 23.541999999999998
    - type: precision_at_10
      value: 7.269
    - type: precision_at_100
      value: 0.9690000000000001
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 15.979
    - type: precision_at_5
      value: 11.664
    - type: recall_at_1
      value: 23.541999999999998
    - type: recall_at_10
      value: 72.688
    - type: recall_at_100
      value: 96.871
    - type: recall_at_1000
      value: 99.431
    - type: recall_at_3
      value: 47.937000000000005
    - type: recall_at_5
      value: 58.321
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
      value: 45.546499570522094
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
      value: 41.01607489943561
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
      value: 59.616107510107774
    - type: mrr
      value: 72.75106626214661
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
      value: 84.33018094733868
    - type: cos_sim_spearman
      value: 83.60190492611737
    - type: euclidean_pearson
      value: 82.1492450218961
    - type: euclidean_spearman
      value: 82.70308926526991
    - type: manhattan_pearson
      value: 81.93959600076842
    - type: manhattan_spearman
      value: 82.73260801016369
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
      value: 84.54545454545455
    - type: f1
      value: 84.49582530928923
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
      value: 37.362725540120096
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
      value: 34.849509608178145
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
      value: 31.502999999999997
    - type: map_at_10
      value: 43.323
    - type: map_at_100
      value: 44.708999999999996
    - type: map_at_1000
      value: 44.838
    - type: map_at_3
      value: 38.987
    - type: map_at_5
      value: 41.516999999999996
    - type: mrr_at_1
      value: 38.769999999999996
    - type: mrr_at_10
      value: 49.13
    - type: mrr_at_100
      value: 49.697
    - type: mrr_at_1000
      value: 49.741
    - type: mrr_at_3
      value: 45.804
    - type: mrr_at_5
      value: 47.842
    - type: ndcg_at_1
      value: 38.769999999999996
    - type: ndcg_at_10
      value: 50.266999999999996
    - type: ndcg_at_100
      value: 54.967
    - type: ndcg_at_1000
      value: 56.976000000000006
    - type: ndcg_at_3
      value: 43.823
    - type: ndcg_at_5
      value: 47.12
    - type: precision_at_1
      value: 38.769999999999996
    - type: precision_at_10
      value: 10.057
    - type: precision_at_100
      value: 1.554
    - type: precision_at_1000
      value: 0.202
    - type: precision_at_3
      value: 21.125
    - type: precision_at_5
      value: 15.851
    - type: recall_at_1
      value: 31.502999999999997
    - type: recall_at_10
      value: 63.715999999999994
    - type: recall_at_100
      value: 83.61800000000001
    - type: recall_at_1000
      value: 96.63199999999999
    - type: recall_at_3
      value: 45.403
    - type: recall_at_5
      value: 54.481
    - type: map_at_1
      value: 27.833000000000002
    - type: map_at_10
      value: 37.330999999999996
    - type: map_at_100
      value: 38.580999999999996
    - type: map_at_1000
      value: 38.708
    - type: map_at_3
      value: 34.713
    - type: map_at_5
      value: 36.104
    - type: mrr_at_1
      value: 35.223
    - type: mrr_at_10
      value: 43.419000000000004
    - type: mrr_at_100
      value: 44.198
    - type: mrr_at_1000
      value: 44.249
    - type: mrr_at_3
      value: 41.614000000000004
    - type: mrr_at_5
      value: 42.553000000000004
    - type: ndcg_at_1
      value: 35.223
    - type: ndcg_at_10
      value: 42.687999999999995
    - type: ndcg_at_100
      value: 47.447
    - type: ndcg_at_1000
      value: 49.701
    - type: ndcg_at_3
      value: 39.162
    - type: ndcg_at_5
      value: 40.557
    - type: precision_at_1
      value: 35.223
    - type: precision_at_10
      value: 7.962
    - type: precision_at_100
      value: 1.304
    - type: precision_at_1000
      value: 0.18
    - type: precision_at_3
      value: 19.023
    - type: precision_at_5
      value: 13.184999999999999
    - type: recall_at_1
      value: 27.833000000000002
    - type: recall_at_10
      value: 51.881
    - type: recall_at_100
      value: 72.04
    - type: recall_at_1000
      value: 86.644
    - type: recall_at_3
      value: 40.778
    - type: recall_at_5
      value: 45.176
    - type: map_at_1
      value: 38.175
    - type: map_at_10
      value: 51.174
    - type: map_at_100
      value: 52.26499999999999
    - type: map_at_1000
      value: 52.315999999999995
    - type: map_at_3
      value: 47.897
    - type: map_at_5
      value: 49.703
    - type: mrr_at_1
      value: 43.448
    - type: mrr_at_10
      value: 54.505
    - type: mrr_at_100
      value: 55.216
    - type: mrr_at_1000
      value: 55.242000000000004
    - type: mrr_at_3
      value: 51.98500000000001
    - type: mrr_at_5
      value: 53.434000000000005
    - type: ndcg_at_1
      value: 43.448
    - type: ndcg_at_10
      value: 57.282
    - type: ndcg_at_100
      value: 61.537
    - type: ndcg_at_1000
      value: 62.546
    - type: ndcg_at_3
      value: 51.73799999999999
    - type: ndcg_at_5
      value: 54.324
    - type: precision_at_1
      value: 43.448
    - type: precision_at_10
      value: 9.292
    - type: precision_at_100
      value: 1.233
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 23.218
    - type: precision_at_5
      value: 15.887
    - type: recall_at_1
      value: 38.175
    - type: recall_at_10
      value: 72.00999999999999
    - type: recall_at_100
      value: 90.155
    - type: recall_at_1000
      value: 97.257
    - type: recall_at_3
      value: 57.133
    - type: recall_at_5
      value: 63.424
    - type: map_at_1
      value: 22.405
    - type: map_at_10
      value: 30.043
    - type: map_at_100
      value: 31.191000000000003
    - type: map_at_1000
      value: 31.275
    - type: map_at_3
      value: 27.034000000000002
    - type: map_at_5
      value: 28.688000000000002
    - type: mrr_at_1
      value: 24.068
    - type: mrr_at_10
      value: 31.993
    - type: mrr_at_100
      value: 32.992
    - type: mrr_at_1000
      value: 33.050000000000004
    - type: mrr_at_3
      value: 28.964000000000002
    - type: mrr_at_5
      value: 30.653000000000002
    - type: ndcg_at_1
      value: 24.068
    - type: ndcg_at_10
      value: 35.198
    - type: ndcg_at_100
      value: 40.709
    - type: ndcg_at_1000
      value: 42.855
    - type: ndcg_at_3
      value: 29.139
    - type: ndcg_at_5
      value: 32.045
    - type: precision_at_1
      value: 24.068
    - type: precision_at_10
      value: 5.65
    - type: precision_at_100
      value: 0.885
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 12.279
    - type: precision_at_5
      value: 8.994
    - type: recall_at_1
      value: 22.405
    - type: recall_at_10
      value: 49.391
    - type: recall_at_100
      value: 74.53699999999999
    - type: recall_at_1000
      value: 90.605
    - type: recall_at_3
      value: 33.126
    - type: recall_at_5
      value: 40.073
    - type: map_at_1
      value: 13.309999999999999
    - type: map_at_10
      value: 20.688000000000002
    - type: map_at_100
      value: 22.022
    - type: map_at_1000
      value: 22.152
    - type: map_at_3
      value: 17.954
    - type: map_at_5
      value: 19.439
    - type: mrr_at_1
      value: 16.294
    - type: mrr_at_10
      value: 24.479
    - type: mrr_at_100
      value: 25.515
    - type: mrr_at_1000
      value: 25.593
    - type: mrr_at_3
      value: 21.642
    - type: mrr_at_5
      value: 23.189999999999998
    - type: ndcg_at_1
      value: 16.294
    - type: ndcg_at_10
      value: 25.833000000000002
    - type: ndcg_at_100
      value: 32.074999999999996
    - type: ndcg_at_1000
      value: 35.083
    - type: ndcg_at_3
      value: 20.493
    - type: ndcg_at_5
      value: 22.949
    - type: precision_at_1
      value: 16.294
    - type: precision_at_10
      value: 5.112
    - type: precision_at_100
      value: 0.96
    - type: precision_at_1000
      value: 0.134
    - type: precision_at_3
      value: 9.908999999999999
    - type: precision_at_5
      value: 7.587000000000001
    - type: recall_at_1
      value: 13.309999999999999
    - type: recall_at_10
      value: 37.851
    - type: recall_at_100
      value: 64.835
    - type: recall_at_1000
      value: 86.334
    - type: recall_at_3
      value: 23.493
    - type: recall_at_5
      value: 29.528
    - type: map_at_1
      value: 25.857999999999997
    - type: map_at_10
      value: 35.503
    - type: map_at_100
      value: 36.957
    - type: map_at_1000
      value: 37.065
    - type: map_at_3
      value: 32.275999999999996
    - type: map_at_5
      value: 34.119
    - type: mrr_at_1
      value: 31.954
    - type: mrr_at_10
      value: 40.851
    - type: mrr_at_100
      value: 41.863
    - type: mrr_at_1000
      value: 41.900999999999996
    - type: mrr_at_3
      value: 38.129999999999995
    - type: mrr_at_5
      value: 39.737
    - type: ndcg_at_1
      value: 31.954
    - type: ndcg_at_10
      value: 41.343999999999994
    - type: ndcg_at_100
      value: 47.397
    - type: ndcg_at_1000
      value: 49.501
    - type: ndcg_at_3
      value: 36.047000000000004
    - type: ndcg_at_5
      value: 38.639
    - type: precision_at_1
      value: 31.954
    - type: precision_at_10
      value: 7.68
    - type: precision_at_100
      value: 1.247
    - type: precision_at_1000
      value: 0.16199999999999998
    - type: precision_at_3
      value: 17.132
    - type: precision_at_5
      value: 12.589
    - type: recall_at_1
      value: 25.857999999999997
    - type: recall_at_10
      value: 53.43599999999999
    - type: recall_at_100
      value: 78.82400000000001
    - type: recall_at_1000
      value: 92.78999999999999
    - type: recall_at_3
      value: 38.655
    - type: recall_at_5
      value: 45.216
    - type: map_at_1
      value: 24.709
    - type: map_at_10
      value: 34.318
    - type: map_at_100
      value: 35.657
    - type: map_at_1000
      value: 35.783
    - type: map_at_3
      value: 31.326999999999998
    - type: map_at_5
      value: 33.021
    - type: mrr_at_1
      value: 30.137000000000004
    - type: mrr_at_10
      value: 39.093
    - type: mrr_at_100
      value: 39.992
    - type: mrr_at_1000
      value: 40.056999999999995
    - type: mrr_at_3
      value: 36.606
    - type: mrr_at_5
      value: 37.861
    - type: ndcg_at_1
      value: 30.137000000000004
    - type: ndcg_at_10
      value: 39.974
    - type: ndcg_at_100
      value: 45.647999999999996
    - type: ndcg_at_1000
      value: 48.259
    - type: ndcg_at_3
      value: 35.028
    - type: ndcg_at_5
      value: 37.175999999999995
    - type: precision_at_1
      value: 30.137000000000004
    - type: precision_at_10
      value: 7.363
    - type: precision_at_100
      value: 1.184
    - type: precision_at_1000
      value: 0.161
    - type: precision_at_3
      value: 16.857
    - type: precision_at_5
      value: 11.963
    - type: recall_at_1
      value: 24.709
    - type: recall_at_10
      value: 52.087
    - type: recall_at_100
      value: 76.125
    - type: recall_at_1000
      value: 93.82300000000001
    - type: recall_at_3
      value: 38.149
    - type: recall_at_5
      value: 43.984
    - type: map_at_1
      value: 23.40791666666667
    - type: map_at_10
      value: 32.458083333333335
    - type: map_at_100
      value: 33.691916666666664
    - type: map_at_1000
      value: 33.81191666666666
    - type: map_at_3
      value: 29.51625
    - type: map_at_5
      value: 31.168083333333335
    - type: mrr_at_1
      value: 27.96591666666666
    - type: mrr_at_10
      value: 36.528583333333344
    - type: mrr_at_100
      value: 37.404
    - type: mrr_at_1000
      value: 37.464333333333336
    - type: mrr_at_3
      value: 33.92883333333333
    - type: mrr_at_5
      value: 35.41933333333333
    - type: ndcg_at_1
      value: 27.96591666666666
    - type: ndcg_at_10
      value: 37.89141666666666
    - type: ndcg_at_100
      value: 43.23066666666666
    - type: ndcg_at_1000
      value: 45.63258333333333
    - type: ndcg_at_3
      value: 32.811249999999994
    - type: ndcg_at_5
      value: 35.22566666666667
    - type: precision_at_1
      value: 27.96591666666666
    - type: precision_at_10
      value: 6.834083333333332
    - type: precision_at_100
      value: 1.12225
    - type: precision_at_1000
      value: 0.15241666666666667
    - type: precision_at_3
      value: 15.264333333333335
    - type: precision_at_5
      value: 11.039416666666666
    - type: recall_at_1
      value: 23.40791666666667
    - type: recall_at_10
      value: 49.927083333333336
    - type: recall_at_100
      value: 73.44641666666668
    - type: recall_at_1000
      value: 90.19950000000001
    - type: recall_at_3
      value: 35.88341666666667
    - type: recall_at_5
      value: 42.061249999999994
    - type: map_at_1
      value: 19.592000000000002
    - type: map_at_10
      value: 26.895999999999997
    - type: map_at_100
      value: 27.921000000000003
    - type: map_at_1000
      value: 28.02
    - type: map_at_3
      value: 24.883
    - type: map_at_5
      value: 25.812
    - type: mrr_at_1
      value: 22.698999999999998
    - type: mrr_at_10
      value: 29.520999999999997
    - type: mrr_at_100
      value: 30.458000000000002
    - type: mrr_at_1000
      value: 30.526999999999997
    - type: mrr_at_3
      value: 27.633000000000003
    - type: mrr_at_5
      value: 28.483999999999998
    - type: ndcg_at_1
      value: 22.698999999999998
    - type: ndcg_at_10
      value: 31.061
    - type: ndcg_at_100
      value: 36.398
    - type: ndcg_at_1000
      value: 38.89
    - type: ndcg_at_3
      value: 27.149
    - type: ndcg_at_5
      value: 28.627000000000002
    - type: precision_at_1
      value: 22.698999999999998
    - type: precision_at_10
      value: 5.106999999999999
    - type: precision_at_100
      value: 0.857
    - type: precision_at_1000
      value: 0.11499999999999999
    - type: precision_at_3
      value: 11.963
    - type: precision_at_5
      value: 8.221
    - type: recall_at_1
      value: 19.592000000000002
    - type: recall_at_10
      value: 41.329
    - type: recall_at_100
      value: 66.094
    - type: recall_at_1000
      value: 84.511
    - type: recall_at_3
      value: 30.61
    - type: recall_at_5
      value: 34.213
    - type: map_at_1
      value: 14.71
    - type: map_at_10
      value: 20.965
    - type: map_at_100
      value: 21.994
    - type: map_at_1000
      value: 22.133
    - type: map_at_3
      value: 18.741
    - type: map_at_5
      value: 19.951
    - type: mrr_at_1
      value: 18.307000000000002
    - type: mrr_at_10
      value: 24.66
    - type: mrr_at_100
      value: 25.540000000000003
    - type: mrr_at_1000
      value: 25.629
    - type: mrr_at_3
      value: 22.511
    - type: mrr_at_5
      value: 23.72
    - type: ndcg_at_1
      value: 18.307000000000002
    - type: ndcg_at_10
      value: 25.153
    - type: ndcg_at_100
      value: 30.229
    - type: ndcg_at_1000
      value: 33.623
    - type: ndcg_at_3
      value: 21.203
    - type: ndcg_at_5
      value: 23.006999999999998
    - type: precision_at_1
      value: 18.307000000000002
    - type: precision_at_10
      value: 4.725
    - type: precision_at_100
      value: 0.8659999999999999
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 10.14
    - type: precision_at_5
      value: 7.481
    - type: recall_at_1
      value: 14.71
    - type: recall_at_10
      value: 34.087
    - type: recall_at_100
      value: 57.147999999999996
    - type: recall_at_1000
      value: 81.777
    - type: recall_at_3
      value: 22.996
    - type: recall_at_5
      value: 27.73
    - type: map_at_1
      value: 23.472
    - type: map_at_10
      value: 32.699
    - type: map_at_100
      value: 33.867000000000004
    - type: map_at_1000
      value: 33.967000000000006
    - type: map_at_3
      value: 29.718
    - type: map_at_5
      value: 31.345
    - type: mrr_at_1
      value: 28.265
    - type: mrr_at_10
      value: 36.945
    - type: mrr_at_100
      value: 37.794
    - type: mrr_at_1000
      value: 37.857
    - type: mrr_at_3
      value: 34.266000000000005
    - type: mrr_at_5
      value: 35.768
    - type: ndcg_at_1
      value: 28.265
    - type: ndcg_at_10
      value: 38.35
    - type: ndcg_at_100
      value: 43.739
    - type: ndcg_at_1000
      value: 46.087
    - type: ndcg_at_3
      value: 33.004
    - type: ndcg_at_5
      value: 35.411
    - type: precision_at_1
      value: 28.265
    - type: precision_at_10
      value: 6.715999999999999
    - type: precision_at_100
      value: 1.059
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 15.299
    - type: precision_at_5
      value: 10.951
    - type: recall_at_1
      value: 23.472
    - type: recall_at_10
      value: 51.413
    - type: recall_at_100
      value: 75.17
    - type: recall_at_1000
      value: 91.577
    - type: recall_at_3
      value: 36.651
    - type: recall_at_5
      value: 42.814
    - type: map_at_1
      value: 23.666
    - type: map_at_10
      value: 32.963
    - type: map_at_100
      value: 34.544999999999995
    - type: map_at_1000
      value: 34.792
    - type: map_at_3
      value: 29.74
    - type: map_at_5
      value: 31.5
    - type: mrr_at_1
      value: 29.051
    - type: mrr_at_10
      value: 38.013000000000005
    - type: mrr_at_100
      value: 38.997
    - type: mrr_at_1000
      value: 39.055
    - type: mrr_at_3
      value: 34.947
    - type: mrr_at_5
      value: 36.815
    - type: ndcg_at_1
      value: 29.051
    - type: ndcg_at_10
      value: 39.361000000000004
    - type: ndcg_at_100
      value: 45.186
    - type: ndcg_at_1000
      value: 47.867
    - type: ndcg_at_3
      value: 33.797
    - type: ndcg_at_5
      value: 36.456
    - type: precision_at_1
      value: 29.051
    - type: precision_at_10
      value: 7.668
    - type: precision_at_100
      value: 1.532
    - type: precision_at_1000
      value: 0.247
    - type: precision_at_3
      value: 15.876000000000001
    - type: precision_at_5
      value: 11.779
    - type: recall_at_1
      value: 23.666
    - type: recall_at_10
      value: 51.858000000000004
    - type: recall_at_100
      value: 77.805
    - type: recall_at_1000
      value: 94.504
    - type: recall_at_3
      value: 36.207
    - type: recall_at_5
      value: 43.094
    - type: map_at_1
      value: 15.662
    - type: map_at_10
      value: 23.594
    - type: map_at_100
      value: 24.593999999999998
    - type: map_at_1000
      value: 24.694
    - type: map_at_3
      value: 20.925
    - type: map_at_5
      value: 22.817999999999998
    - type: mrr_at_1
      value: 17.375
    - type: mrr_at_10
      value: 25.734
    - type: mrr_at_100
      value: 26.586
    - type: mrr_at_1000
      value: 26.671
    - type: mrr_at_3
      value: 23.044
    - type: mrr_at_5
      value: 24.975
    - type: ndcg_at_1
      value: 17.375
    - type: ndcg_at_10
      value: 28.186
    - type: ndcg_at_100
      value: 33.436
    - type: ndcg_at_1000
      value: 36.203
    - type: ndcg_at_3
      value: 23.152
    - type: ndcg_at_5
      value: 26.397
    - type: precision_at_1
      value: 17.375
    - type: precision_at_10
      value: 4.677
    - type: precision_at_100
      value: 0.786
    - type: precision_at_1000
      value: 0.109
    - type: precision_at_3
      value: 10.351
    - type: precision_at_5
      value: 7.985
    - type: recall_at_1
      value: 15.662
    - type: recall_at_10
      value: 40.066
    - type: recall_at_100
      value: 65.006
    - type: recall_at_1000
      value: 85.94000000000001
    - type: recall_at_3
      value: 27.400000000000002
    - type: recall_at_5
      value: 35.002
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
      value: 8.853
    - type: map_at_10
      value: 15.568000000000001
    - type: map_at_100
      value: 17.383000000000003
    - type: map_at_1000
      value: 17.584
    - type: map_at_3
      value: 12.561
    - type: map_at_5
      value: 14.056
    - type: mrr_at_1
      value: 18.958
    - type: mrr_at_10
      value: 28.288000000000004
    - type: mrr_at_100
      value: 29.432000000000002
    - type: mrr_at_1000
      value: 29.498
    - type: mrr_at_3
      value: 25.049
    - type: mrr_at_5
      value: 26.857
    - type: ndcg_at_1
      value: 18.958
    - type: ndcg_at_10
      value: 22.21
    - type: ndcg_at_100
      value: 29.596
    - type: ndcg_at_1000
      value: 33.583
    - type: ndcg_at_3
      value: 16.994999999999997
    - type: ndcg_at_5
      value: 18.95
    - type: precision_at_1
      value: 18.958
    - type: precision_at_10
      value: 7.192
    - type: precision_at_100
      value: 1.5
    - type: precision_at_1000
      value: 0.22399999999999998
    - type: precision_at_3
      value: 12.573
    - type: precision_at_5
      value: 10.202
    - type: recall_at_1
      value: 8.853
    - type: recall_at_10
      value: 28.087
    - type: recall_at_100
      value: 53.701
    - type: recall_at_1000
      value: 76.29899999999999
    - type: recall_at_3
      value: 15.913
    - type: recall_at_5
      value: 20.658
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
      value: 9.077
    - type: map_at_10
      value: 20.788999999999998
    - type: map_at_100
      value: 30.429000000000002
    - type: map_at_1000
      value: 32.143
    - type: map_at_3
      value: 14.692
    - type: map_at_5
      value: 17.139
    - type: mrr_at_1
      value: 70.75
    - type: mrr_at_10
      value: 78.036
    - type: mrr_at_100
      value: 78.401
    - type: mrr_at_1000
      value: 78.404
    - type: mrr_at_3
      value: 76.75
    - type: mrr_at_5
      value: 77.47500000000001
    - type: ndcg_at_1
      value: 58.12500000000001
    - type: ndcg_at_10
      value: 44.015
    - type: ndcg_at_100
      value: 49.247
    - type: ndcg_at_1000
      value: 56.211999999999996
    - type: ndcg_at_3
      value: 49.151
    - type: ndcg_at_5
      value: 46.195
    - type: precision_at_1
      value: 70.75
    - type: precision_at_10
      value: 35.5
    - type: precision_at_100
      value: 11.355
    - type: precision_at_1000
      value: 2.1950000000000003
    - type: precision_at_3
      value: 53.083000000000006
    - type: precision_at_5
      value: 44.800000000000004
    - type: recall_at_1
      value: 9.077
    - type: recall_at_10
      value: 26.259
    - type: recall_at_100
      value: 56.547000000000004
    - type: recall_at_1000
      value: 78.551
    - type: recall_at_3
      value: 16.162000000000003
    - type: recall_at_5
      value: 19.753999999999998
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
      value: 49.44500000000001
    - type: f1
      value: 44.67067691783401
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
      value: 68.182
    - type: map_at_10
      value: 78.223
    - type: map_at_100
      value: 78.498
    - type: map_at_1000
      value: 78.512
    - type: map_at_3
      value: 76.71
    - type: map_at_5
      value: 77.725
    - type: mrr_at_1
      value: 73.177
    - type: mrr_at_10
      value: 82.513
    - type: mrr_at_100
      value: 82.633
    - type: mrr_at_1000
      value: 82.635
    - type: mrr_at_3
      value: 81.376
    - type: mrr_at_5
      value: 82.182
    - type: ndcg_at_1
      value: 73.177
    - type: ndcg_at_10
      value: 82.829
    - type: ndcg_at_100
      value: 83.84
    - type: ndcg_at_1000
      value: 84.07900000000001
    - type: ndcg_at_3
      value: 80.303
    - type: ndcg_at_5
      value: 81.846
    - type: precision_at_1
      value: 73.177
    - type: precision_at_10
      value: 10.241999999999999
    - type: precision_at_100
      value: 1.099
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 31.247999999999998
    - type: precision_at_5
      value: 19.697
    - type: recall_at_1
      value: 68.182
    - type: recall_at_10
      value: 92.657
    - type: recall_at_100
      value: 96.709
    - type: recall_at_1000
      value: 98.184
    - type: recall_at_3
      value: 85.9
    - type: recall_at_5
      value: 89.755
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
      value: 21.108
    - type: map_at_10
      value: 33.342
    - type: map_at_100
      value: 35.281
    - type: map_at_1000
      value: 35.478
    - type: map_at_3
      value: 29.067
    - type: map_at_5
      value: 31.563000000000002
    - type: mrr_at_1
      value: 41.667
    - type: mrr_at_10
      value: 49.913000000000004
    - type: mrr_at_100
      value: 50.724000000000004
    - type: mrr_at_1000
      value: 50.766
    - type: mrr_at_3
      value: 47.504999999999995
    - type: mrr_at_5
      value: 49.033
    - type: ndcg_at_1
      value: 41.667
    - type: ndcg_at_10
      value: 41.144
    - type: ndcg_at_100
      value: 48.326
    - type: ndcg_at_1000
      value: 51.486
    - type: ndcg_at_3
      value: 37.486999999999995
    - type: ndcg_at_5
      value: 38.78
    - type: precision_at_1
      value: 41.667
    - type: precision_at_10
      value: 11.358
    - type: precision_at_100
      value: 1.873
    - type: precision_at_1000
      value: 0.244
    - type: precision_at_3
      value: 25
    - type: precision_at_5
      value: 18.519
    - type: recall_at_1
      value: 21.108
    - type: recall_at_10
      value: 47.249
    - type: recall_at_100
      value: 74.52
    - type: recall_at_1000
      value: 93.31
    - type: recall_at_3
      value: 33.271
    - type: recall_at_5
      value: 39.723000000000006
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
      value: 40.317
    - type: map_at_10
      value: 64.861
    - type: map_at_100
      value: 65.697
    - type: map_at_1000
      value: 65.755
    - type: map_at_3
      value: 61.258
    - type: map_at_5
      value: 63.590999999999994
    - type: mrr_at_1
      value: 80.635
    - type: mrr_at_10
      value: 86.528
    - type: mrr_at_100
      value: 86.66199999999999
    - type: mrr_at_1000
      value: 86.666
    - type: mrr_at_3
      value: 85.744
    - type: mrr_at_5
      value: 86.24300000000001
    - type: ndcg_at_1
      value: 80.635
    - type: ndcg_at_10
      value: 73.13199999999999
    - type: ndcg_at_100
      value: 75.927
    - type: ndcg_at_1000
      value: 76.976
    - type: ndcg_at_3
      value: 68.241
    - type: ndcg_at_5
      value: 71.071
    - type: precision_at_1
      value: 80.635
    - type: precision_at_10
      value: 15.326
    - type: precision_at_100
      value: 1.7500000000000002
    - type: precision_at_1000
      value: 0.189
    - type: precision_at_3
      value: 43.961
    - type: precision_at_5
      value: 28.599999999999998
    - type: recall_at_1
      value: 40.317
    - type: recall_at_10
      value: 76.631
    - type: recall_at_100
      value: 87.495
    - type: recall_at_1000
      value: 94.362
    - type: recall_at_3
      value: 65.94200000000001
    - type: recall_at_5
      value: 71.499
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
      value: 91.686
    - type: ap
      value: 87.5577120393173
    - type: f1
      value: 91.6629447355139
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
      value: 23.702
    - type: map_at_10
      value: 36.414
    - type: map_at_100
      value: 37.561
    - type: map_at_1000
      value: 37.605
    - type: map_at_3
      value: 32.456
    - type: map_at_5
      value: 34.827000000000005
    - type: mrr_at_1
      value: 24.355
    - type: mrr_at_10
      value: 37.01
    - type: mrr_at_100
      value: 38.085
    - type: mrr_at_1000
      value: 38.123000000000005
    - type: mrr_at_3
      value: 33.117999999999995
    - type: mrr_at_5
      value: 35.452
    - type: ndcg_at_1
      value: 24.384
    - type: ndcg_at_10
      value: 43.456
    - type: ndcg_at_100
      value: 48.892
    - type: ndcg_at_1000
      value: 49.964
    - type: ndcg_at_3
      value: 35.475
    - type: ndcg_at_5
      value: 39.711
    - type: precision_at_1
      value: 24.384
    - type: precision_at_10
      value: 6.7940000000000005
    - type: precision_at_100
      value: 0.951
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 15.052999999999999
    - type: precision_at_5
      value: 11.189
    - type: recall_at_1
      value: 23.702
    - type: recall_at_10
      value: 65.057
    - type: recall_at_100
      value: 90.021
    - type: recall_at_1000
      value: 98.142
    - type: recall_at_3
      value: 43.551
    - type: recall_at_5
      value: 53.738
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
      value: 94.62380300957591
    - type: f1
      value: 94.49871222100734
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
      value: 77.14090287277702
    - type: f1
      value: 60.32101258220515
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
      value: 73.84330867518494
    - type: f1
      value: 71.92248688515255
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
      value: 78.10692669804976
    - type: f1
      value: 77.9904839122866
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
      value: 31.822988923078444
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
      value: 30.38394880253403
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
      value: 31.82504612539082
    - type: mrr
      value: 32.84462298174977
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
      value: 6.029
    - type: map_at_10
      value: 14.088999999999999
    - type: map_at_100
      value: 17.601
    - type: map_at_1000
      value: 19.144
    - type: map_at_3
      value: 10.156
    - type: map_at_5
      value: 11.892
    - type: mrr_at_1
      value: 46.44
    - type: mrr_at_10
      value: 56.596999999999994
    - type: mrr_at_100
      value: 57.11000000000001
    - type: mrr_at_1000
      value: 57.14
    - type: mrr_at_3
      value: 54.334
    - type: mrr_at_5
      value: 55.774
    - type: ndcg_at_1
      value: 44.891999999999996
    - type: ndcg_at_10
      value: 37.134
    - type: ndcg_at_100
      value: 33.652
    - type: ndcg_at_1000
      value: 42.548
    - type: ndcg_at_3
      value: 41.851
    - type: ndcg_at_5
      value: 39.842
    - type: precision_at_1
      value: 46.44
    - type: precision_at_10
      value: 27.647
    - type: precision_at_100
      value: 8.309999999999999
    - type: precision_at_1000
      value: 2.146
    - type: precision_at_3
      value: 39.422000000000004
    - type: precision_at_5
      value: 34.675
    - type: recall_at_1
      value: 6.029
    - type: recall_at_10
      value: 18.907
    - type: recall_at_100
      value: 33.76
    - type: recall_at_1000
      value: 65.14999999999999
    - type: recall_at_3
      value: 11.584999999999999
    - type: recall_at_5
      value: 14.626
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
      value: 39.373000000000005
    - type: map_at_10
      value: 55.836
    - type: map_at_100
      value: 56.611999999999995
    - type: map_at_1000
      value: 56.63
    - type: map_at_3
      value: 51.747
    - type: map_at_5
      value: 54.337999999999994
    - type: mrr_at_1
      value: 44.147999999999996
    - type: mrr_at_10
      value: 58.42699999999999
    - type: mrr_at_100
      value: 58.902
    - type: mrr_at_1000
      value: 58.914
    - type: mrr_at_3
      value: 55.156000000000006
    - type: mrr_at_5
      value: 57.291000000000004
    - type: ndcg_at_1
      value: 44.119
    - type: ndcg_at_10
      value: 63.444
    - type: ndcg_at_100
      value: 66.40599999999999
    - type: ndcg_at_1000
      value: 66.822
    - type: ndcg_at_3
      value: 55.962
    - type: ndcg_at_5
      value: 60.228
    - type: precision_at_1
      value: 44.119
    - type: precision_at_10
      value: 10.006
    - type: precision_at_100
      value: 1.17
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 25.135
    - type: precision_at_5
      value: 17.59
    - type: recall_at_1
      value: 39.373000000000005
    - type: recall_at_10
      value: 83.78999999999999
    - type: recall_at_100
      value: 96.246
    - type: recall_at_1000
      value: 99.324
    - type: recall_at_3
      value: 64.71900000000001
    - type: recall_at_5
      value: 74.508
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
      value: 69.199
    - type: map_at_10
      value: 82.892
    - type: map_at_100
      value: 83.578
    - type: map_at_1000
      value: 83.598
    - type: map_at_3
      value: 79.948
    - type: map_at_5
      value: 81.779
    - type: mrr_at_1
      value: 79.67
    - type: mrr_at_10
      value: 86.115
    - type: mrr_at_100
      value: 86.249
    - type: mrr_at_1000
      value: 86.251
    - type: mrr_at_3
      value: 85.08200000000001
    - type: mrr_at_5
      value: 85.783
    - type: ndcg_at_1
      value: 79.67
    - type: ndcg_at_10
      value: 86.839
    - type: ndcg_at_100
      value: 88.252
    - type: ndcg_at_1000
      value: 88.401
    - type: ndcg_at_3
      value: 83.86200000000001
    - type: ndcg_at_5
      value: 85.473
    - type: precision_at_1
      value: 79.67
    - type: precision_at_10
      value: 13.19
    - type: precision_at_100
      value: 1.521
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 36.677
    - type: precision_at_5
      value: 24.118000000000002
    - type: recall_at_1
      value: 69.199
    - type: recall_at_10
      value: 94.321
    - type: recall_at_100
      value: 99.20400000000001
    - type: recall_at_1000
      value: 99.947
    - type: recall_at_3
      value: 85.787
    - type: recall_at_5
      value: 90.365
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
      value: 55.82810046856353
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
      value: 63.38132611783628
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
      value: 5.127000000000001
    - type: map_at_10
      value: 12.235
    - type: map_at_100
      value: 14.417
    - type: map_at_1000
      value: 14.75
    - type: map_at_3
      value: 8.906
    - type: map_at_5
      value: 10.591000000000001
    - type: mrr_at_1
      value: 25.2
    - type: mrr_at_10
      value: 35.879
    - type: mrr_at_100
      value: 36.935
    - type: mrr_at_1000
      value: 36.997
    - type: mrr_at_3
      value: 32.783
    - type: mrr_at_5
      value: 34.367999999999995
    - type: ndcg_at_1
      value: 25.2
    - type: ndcg_at_10
      value: 20.509
    - type: ndcg_at_100
      value: 28.67
    - type: ndcg_at_1000
      value: 34.42
    - type: ndcg_at_3
      value: 19.948
    - type: ndcg_at_5
      value: 17.166
    - type: precision_at_1
      value: 25.2
    - type: precision_at_10
      value: 10.440000000000001
    - type: precision_at_100
      value: 2.214
    - type: precision_at_1000
      value: 0.359
    - type: precision_at_3
      value: 18.533
    - type: precision_at_5
      value: 14.860000000000001
    - type: recall_at_1
      value: 5.127000000000001
    - type: recall_at_10
      value: 21.147
    - type: recall_at_100
      value: 44.946999999999996
    - type: recall_at_1000
      value: 72.89
    - type: recall_at_3
      value: 11.277
    - type: recall_at_5
      value: 15.042
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
      value: 83.0373011786213
    - type: cos_sim_spearman
      value: 79.27889560856613
    - type: euclidean_pearson
      value: 80.31186315495655
    - type: euclidean_spearman
      value: 79.41630415280811
    - type: manhattan_pearson
      value: 80.31755140442013
    - type: manhattan_spearman
      value: 79.43069870027611
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
      value: 84.8659751342045
    - type: cos_sim_spearman
      value: 76.95377612997667
    - type: euclidean_pearson
      value: 81.24552945497848
    - type: euclidean_spearman
      value: 77.18236963555253
    - type: manhattan_pearson
      value: 81.26477607759037
    - type: manhattan_spearman
      value: 77.13821753062756
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
      value: 83.34597139044875
    - type: cos_sim_spearman
      value: 84.124169425592
    - type: euclidean_pearson
      value: 83.68590721511401
    - type: euclidean_spearman
      value: 84.18846190846398
    - type: manhattan_pearson
      value: 83.57630235061498
    - type: manhattan_spearman
      value: 84.10244043726902
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
      value: 82.67641885599572
    - type: cos_sim_spearman
      value: 80.46450725650428
    - type: euclidean_pearson
      value: 81.61645042715865
    - type: euclidean_spearman
      value: 80.61418394236874
    - type: manhattan_pearson
      value: 81.55712034928871
    - type: manhattan_spearman
      value: 80.57905670523951
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
      value: 88.86650310886782
    - type: cos_sim_spearman
      value: 89.76081629222328
    - type: euclidean_pearson
      value: 89.1530747029954
    - type: euclidean_spearman
      value: 89.80990657280248
    - type: manhattan_pearson
      value: 89.10640563278132
    - type: manhattan_spearman
      value: 89.76282108434047
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
      value: 83.93864027911118
    - type: cos_sim_spearman
      value: 85.47096193999023
    - type: euclidean_pearson
      value: 85.03141840870533
    - type: euclidean_spearman
      value: 85.43124029598181
    - type: manhattan_pearson
      value: 84.99002664393512
    - type: manhattan_spearman
      value: 85.39169195120834
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
      value: 88.7045343749832
    - type: cos_sim_spearman
      value: 89.03262221146677
    - type: euclidean_pearson
      value: 89.56078218264365
    - type: euclidean_spearman
      value: 89.17827006466868
    - type: manhattan_pearson
      value: 89.52717595468582
    - type: manhattan_spearman
      value: 89.15878115952923
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
      value: 64.20191302875551
    - type: cos_sim_spearman
      value: 64.11446552557646
    - type: euclidean_pearson
      value: 64.6918197393619
    - type: euclidean_spearman
      value: 63.440182631197764
    - type: manhattan_pearson
      value: 64.55692904121835
    - type: manhattan_spearman
      value: 63.424877742756266
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
      value: 86.37793104662344
    - type: cos_sim_spearman
      value: 87.7357802629067
    - type: euclidean_pearson
      value: 87.4286301545109
    - type: euclidean_spearman
      value: 87.78452920777421
    - type: manhattan_pearson
      value: 87.42445169331255
    - type: manhattan_spearman
      value: 87.78537677249598
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
      value: 84.31465405081792
    - type: mrr
      value: 95.7173781193389
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
      value: 57.760999999999996
    - type: map_at_10
      value: 67.904
    - type: map_at_100
      value: 68.539
    - type: map_at_1000
      value: 68.562
    - type: map_at_3
      value: 65.415
    - type: map_at_5
      value: 66.788
    - type: mrr_at_1
      value: 60.333000000000006
    - type: mrr_at_10
      value: 68.797
    - type: mrr_at_100
      value: 69.236
    - type: mrr_at_1000
      value: 69.257
    - type: mrr_at_3
      value: 66.667
    - type: mrr_at_5
      value: 67.967
    - type: ndcg_at_1
      value: 60.333000000000006
    - type: ndcg_at_10
      value: 72.24199999999999
    - type: ndcg_at_100
      value: 74.86
    - type: ndcg_at_1000
      value: 75.354
    - type: ndcg_at_3
      value: 67.93400000000001
    - type: ndcg_at_5
      value: 70.02199999999999
    - type: precision_at_1
      value: 60.333000000000006
    - type: precision_at_10
      value: 9.533
    - type: precision_at_100
      value: 1.09
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 26.778000000000002
    - type: precision_at_5
      value: 17.467
    - type: recall_at_1
      value: 57.760999999999996
    - type: recall_at_10
      value: 84.383
    - type: recall_at_100
      value: 96.267
    - type: recall_at_1000
      value: 100
    - type: recall_at_3
      value: 72.628
    - type: recall_at_5
      value: 78.094
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
      value: 94.9210324173411
    - type: cos_sim_f1
      value: 89.8521162672106
    - type: cos_sim_precision
      value: 91.67533818938605
    - type: cos_sim_recall
      value: 88.1
    - type: dot_accuracy
      value: 99.69504950495049
    - type: dot_ap
      value: 90.4919719146181
    - type: dot_f1
      value: 84.72289156626506
    - type: dot_precision
      value: 81.76744186046511
    - type: dot_recall
      value: 87.9
    - type: euclidean_accuracy
      value: 99.79702970297029
    - type: euclidean_ap
      value: 94.87827463795753
    - type: euclidean_f1
      value: 89.55680081507896
    - type: euclidean_precision
      value: 91.27725856697819
    - type: euclidean_recall
      value: 87.9
    - type: manhattan_accuracy
      value: 99.7990099009901
    - type: manhattan_ap
      value: 94.87587025149682
    - type: manhattan_f1
      value: 89.76298537569339
    - type: manhattan_precision
      value: 90.53916581892166
    - type: manhattan_recall
      value: 89
    - type: max_accuracy
      value: 99.8029702970297
    - type: max_ap
      value: 94.9210324173411
    - type: max_f1
      value: 89.8521162672106
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
      value: 65.92385753948724
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
      value: 33.671756975431144
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
      value: 50.677928036739004
    - type: mrr
      value: 51.56413133435193
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
      value: 30.523589340819683
    - type: cos_sim_spearman
      value: 30.187407518823235
    - type: dot_pearson
      value: 29.039713969699015
    - type: dot_spearman
      value: 29.114740651155508
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
      value: 0.211
    - type: map_at_10
      value: 1.6199999999999999
    - type: map_at_100
      value: 8.658000000000001
    - type: map_at_1000
      value: 21.538
    - type: map_at_3
      value: 0.575
    - type: map_at_5
      value: 0.919
    - type: mrr_at_1
      value: 78
    - type: mrr_at_10
      value: 86.18599999999999
    - type: mrr_at_100
      value: 86.18599999999999
    - type: mrr_at_1000
      value: 86.18599999999999
    - type: mrr_at_3
      value: 85
    - type: mrr_at_5
      value: 85.9
    - type: ndcg_at_1
      value: 74
    - type: ndcg_at_10
      value: 66.542
    - type: ndcg_at_100
      value: 50.163999999999994
    - type: ndcg_at_1000
      value: 45.696999999999996
    - type: ndcg_at_3
      value: 71.531
    - type: ndcg_at_5
      value: 70.45
    - type: precision_at_1
      value: 78
    - type: precision_at_10
      value: 69.39999999999999
    - type: precision_at_100
      value: 51.06
    - type: precision_at_1000
      value: 20.022000000000002
    - type: precision_at_3
      value: 76
    - type: precision_at_5
      value: 74.8
    - type: recall_at_1
      value: 0.211
    - type: recall_at_10
      value: 1.813
    - type: recall_at_100
      value: 12.098
    - type: recall_at_1000
      value: 42.618
    - type: recall_at_3
      value: 0.603
    - type: recall_at_5
      value: 0.987
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
      value: 2.2079999999999997
    - type: map_at_10
      value: 7.777000000000001
    - type: map_at_100
      value: 12.825000000000001
    - type: map_at_1000
      value: 14.196
    - type: map_at_3
      value: 4.285
    - type: map_at_5
      value: 6.177
    - type: mrr_at_1
      value: 30.612000000000002
    - type: mrr_at_10
      value: 42.635
    - type: mrr_at_100
      value: 43.955
    - type: mrr_at_1000
      value: 43.955
    - type: mrr_at_3
      value: 38.435
    - type: mrr_at_5
      value: 41.088
    - type: ndcg_at_1
      value: 28.571
    - type: ndcg_at_10
      value: 20.666999999999998
    - type: ndcg_at_100
      value: 31.840000000000003
    - type: ndcg_at_1000
      value: 43.191
    - type: ndcg_at_3
      value: 23.45
    - type: ndcg_at_5
      value: 22.994
    - type: precision_at_1
      value: 30.612000000000002
    - type: precision_at_10
      value: 17.959
    - type: precision_at_100
      value: 6.755
    - type: precision_at_1000
      value: 1.4200000000000002
    - type: precision_at_3
      value: 23.810000000000002
    - type: precision_at_5
      value: 23.673
    - type: recall_at_1
      value: 2.2079999999999997
    - type: recall_at_10
      value: 13.144
    - type: recall_at_100
      value: 42.491
    - type: recall_at_1000
      value: 77.04299999999999
    - type: recall_at_3
      value: 5.3469999999999995
    - type: recall_at_5
      value: 9.139
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
      value: 70.9044
    - type: ap
      value: 14.625783489340755
    - type: f1
      value: 54.814936562590546
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
      value: 60.94227504244483
    - type: f1
      value: 61.22516038508854
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
      value: 49.602409155145864
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
      value: 86.94641473445789
    - type: cos_sim_ap
      value: 76.91572747061197
    - type: cos_sim_f1
      value: 70.14348097317529
    - type: cos_sim_precision
      value: 66.53254437869822
    - type: cos_sim_recall
      value: 74.1688654353562
    - type: dot_accuracy
      value: 84.80061989628658
    - type: dot_ap
      value: 70.7952548895177
    - type: dot_f1
      value: 65.44780728844965
    - type: dot_precision
      value: 61.53310104529617
    - type: dot_recall
      value: 69.89445910290237
    - type: euclidean_accuracy
      value: 86.94641473445789
    - type: euclidean_ap
      value: 76.80774009393652
    - type: euclidean_f1
      value: 70.30522503879979
    - type: euclidean_precision
      value: 68.94977168949772
    - type: euclidean_recall
      value: 71.71503957783642
    - type: manhattan_accuracy
      value: 86.8629671574179
    - type: manhattan_ap
      value: 76.76518632600317
    - type: manhattan_f1
      value: 70.16056518946692
    - type: manhattan_precision
      value: 68.360450563204
    - type: manhattan_recall
      value: 72.0580474934037
    - type: max_accuracy
      value: 86.94641473445789
    - type: max_ap
      value: 76.91572747061197
    - type: max_f1
      value: 70.30522503879979
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
      value: 89.10428066907285
    - type: cos_sim_ap
      value: 86.25114759921435
    - type: cos_sim_f1
      value: 78.37857884586856
    - type: cos_sim_precision
      value: 75.60818546078993
    - type: cos_sim_recall
      value: 81.35971666153372
    - type: dot_accuracy
      value: 87.41995575736406
    - type: dot_ap
      value: 81.51838010086782
    - type: dot_f1
      value: 74.77398015435503
    - type: dot_precision
      value: 71.53002390662354
    - type: dot_recall
      value: 78.32614721281182
    - type: euclidean_accuracy
      value: 89.12368533395428
    - type: euclidean_ap
      value: 86.33456799874504
    - type: euclidean_f1
      value: 78.45496750232127
    - type: euclidean_precision
      value: 75.78388462366364
    - type: euclidean_recall
      value: 81.32121958731136
    - type: manhattan_accuracy
      value: 89.10622113556099
    - type: manhattan_ap
      value: 86.31215061745333
    - type: manhattan_f1
      value: 78.40684906011539
    - type: manhattan_precision
      value: 75.89536643366722
    - type: manhattan_recall
      value: 81.09023714197721
    - type: max_accuracy
      value: 89.12368533395428
    - type: max_ap
      value: 86.33456799874504
    - type: max_f1
      value: 78.45496750232127
---

# E5-large-v2

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

This model has 24 layers and the embedding size is 1024.

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

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model = AutoModel.from_pretrained('intfloat/e5-large-v2')

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
model = SentenceTransformer('intfloat/e5-large-v2')
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
