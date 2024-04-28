---
language:
- en
license: mit
tags:
- mteb
model-index:
- name: bge-base-en
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
      value: 75.73134328358209
    - type: ap
      value: 38.97277232632892
    - type: f1
      value: 69.81740361139785
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
      value: 92.56522500000001
    - type: ap
      value: 88.88821771869553
    - type: f1
      value: 92.54817512659696
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
      value: 46.91
    - type: f1
      value: 46.28536394320311
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
      value: 38.834
    - type: map_at_10
      value: 53.564
    - type: map_at_100
      value: 54.230000000000004
    - type: map_at_1000
      value: 54.235
    - type: map_at_3
      value: 49.49
    - type: map_at_5
      value: 51.784
    - type: mrr_at_1
      value: 39.26
    - type: mrr_at_10
      value: 53.744
    - type: mrr_at_100
      value: 54.410000000000004
    - type: mrr_at_1000
      value: 54.415
    - type: mrr_at_3
      value: 49.656
    - type: mrr_at_5
      value: 52.018
    - type: ndcg_at_1
      value: 38.834
    - type: ndcg_at_10
      value: 61.487
    - type: ndcg_at_100
      value: 64.303
    - type: ndcg_at_1000
      value: 64.408
    - type: ndcg_at_3
      value: 53.116
    - type: ndcg_at_5
      value: 57.248
    - type: precision_at_1
      value: 38.834
    - type: precision_at_10
      value: 8.663
    - type: precision_at_100
      value: 0.989
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 21.218999999999998
    - type: precision_at_5
      value: 14.737
    - type: recall_at_1
      value: 38.834
    - type: recall_at_10
      value: 86.629
    - type: recall_at_100
      value: 98.86200000000001
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 63.656
    - type: recall_at_5
      value: 73.68400000000001
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
      value: 48.88475477433035
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
      value: 42.85053138403176
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
      value: 62.23221013208242
    - type: mrr
      value: 74.64857318735436
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
      value: 87.4403443247284
    - type: cos_sim_spearman
      value: 85.5326718115169
    - type: euclidean_pearson
      value: 86.0114007449595
    - type: euclidean_spearman
      value: 86.05979225604875
    - type: manhattan_pearson
      value: 86.05423806568598
    - type: manhattan_spearman
      value: 86.02485170086835
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
      value: 86.44480519480518
    - type: f1
      value: 86.41301900941988
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
      value: 40.17547250880036
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
      value: 37.74514172687293
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
      value: 32.096000000000004
    - type: map_at_10
      value: 43.345
    - type: map_at_100
      value: 44.73
    - type: map_at_1000
      value: 44.85
    - type: map_at_3
      value: 39.956
    - type: map_at_5
      value: 41.727
    - type: mrr_at_1
      value: 38.769999999999996
    - type: mrr_at_10
      value: 48.742000000000004
    - type: mrr_at_100
      value: 49.474000000000004
    - type: mrr_at_1000
      value: 49.513
    - type: mrr_at_3
      value: 46.161
    - type: mrr_at_5
      value: 47.721000000000004
    - type: ndcg_at_1
      value: 38.769999999999996
    - type: ndcg_at_10
      value: 49.464999999999996
    - type: ndcg_at_100
      value: 54.632000000000005
    - type: ndcg_at_1000
      value: 56.52
    - type: ndcg_at_3
      value: 44.687
    - type: ndcg_at_5
      value: 46.814
    - type: precision_at_1
      value: 38.769999999999996
    - type: precision_at_10
      value: 9.471
    - type: precision_at_100
      value: 1.4909999999999999
    - type: precision_at_1000
      value: 0.194
    - type: precision_at_3
      value: 21.268
    - type: precision_at_5
      value: 15.079
    - type: recall_at_1
      value: 32.096000000000004
    - type: recall_at_10
      value: 60.99099999999999
    - type: recall_at_100
      value: 83.075
    - type: recall_at_1000
      value: 95.178
    - type: recall_at_3
      value: 47.009
    - type: recall_at_5
      value: 53.348
    - type: map_at_1
      value: 32.588
    - type: map_at_10
      value: 42.251
    - type: map_at_100
      value: 43.478
    - type: map_at_1000
      value: 43.617
    - type: map_at_3
      value: 39.381
    - type: map_at_5
      value: 41.141
    - type: mrr_at_1
      value: 41.21
    - type: mrr_at_10
      value: 48.765
    - type: mrr_at_100
      value: 49.403000000000006
    - type: mrr_at_1000
      value: 49.451
    - type: mrr_at_3
      value: 46.73
    - type: mrr_at_5
      value: 47.965999999999994
    - type: ndcg_at_1
      value: 41.21
    - type: ndcg_at_10
      value: 47.704
    - type: ndcg_at_100
      value: 51.916
    - type: ndcg_at_1000
      value: 54.013999999999996
    - type: ndcg_at_3
      value: 44.007000000000005
    - type: ndcg_at_5
      value: 45.936
    - type: precision_at_1
      value: 41.21
    - type: precision_at_10
      value: 8.885
    - type: precision_at_100
      value: 1.409
    - type: precision_at_1000
      value: 0.189
    - type: precision_at_3
      value: 21.274
    - type: precision_at_5
      value: 15.045
    - type: recall_at_1
      value: 32.588
    - type: recall_at_10
      value: 56.333
    - type: recall_at_100
      value: 74.251
    - type: recall_at_1000
      value: 87.518
    - type: recall_at_3
      value: 44.962
    - type: recall_at_5
      value: 50.609
    - type: map_at_1
      value: 40.308
    - type: map_at_10
      value: 53.12
    - type: map_at_100
      value: 54.123
    - type: map_at_1000
      value: 54.173
    - type: map_at_3
      value: 50.017999999999994
    - type: map_at_5
      value: 51.902
    - type: mrr_at_1
      value: 46.394999999999996
    - type: mrr_at_10
      value: 56.531
    - type: mrr_at_100
      value: 57.19800000000001
    - type: mrr_at_1000
      value: 57.225
    - type: mrr_at_3
      value: 54.368
    - type: mrr_at_5
      value: 55.713
    - type: ndcg_at_1
      value: 46.394999999999996
    - type: ndcg_at_10
      value: 58.811
    - type: ndcg_at_100
      value: 62.834
    - type: ndcg_at_1000
      value: 63.849999999999994
    - type: ndcg_at_3
      value: 53.88699999999999
    - type: ndcg_at_5
      value: 56.477999999999994
    - type: precision_at_1
      value: 46.394999999999996
    - type: precision_at_10
      value: 9.398
    - type: precision_at_100
      value: 1.2309999999999999
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 24.221999999999998
    - type: precision_at_5
      value: 16.539
    - type: recall_at_1
      value: 40.308
    - type: recall_at_10
      value: 72.146
    - type: recall_at_100
      value: 89.60900000000001
    - type: recall_at_1000
      value: 96.733
    - type: recall_at_3
      value: 58.91499999999999
    - type: recall_at_5
      value: 65.34299999999999
    - type: map_at_1
      value: 27.383000000000003
    - type: map_at_10
      value: 35.802
    - type: map_at_100
      value: 36.756
    - type: map_at_1000
      value: 36.826
    - type: map_at_3
      value: 32.923
    - type: map_at_5
      value: 34.577999999999996
    - type: mrr_at_1
      value: 29.604999999999997
    - type: mrr_at_10
      value: 37.918
    - type: mrr_at_100
      value: 38.732
    - type: mrr_at_1000
      value: 38.786
    - type: mrr_at_3
      value: 35.198
    - type: mrr_at_5
      value: 36.808
    - type: ndcg_at_1
      value: 29.604999999999997
    - type: ndcg_at_10
      value: 40.836
    - type: ndcg_at_100
      value: 45.622
    - type: ndcg_at_1000
      value: 47.427
    - type: ndcg_at_3
      value: 35.208
    - type: ndcg_at_5
      value: 38.066
    - type: precision_at_1
      value: 29.604999999999997
    - type: precision_at_10
      value: 6.226
    - type: precision_at_100
      value: 0.9079999999999999
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 14.463000000000001
    - type: precision_at_5
      value: 10.35
    - type: recall_at_1
      value: 27.383000000000003
    - type: recall_at_10
      value: 54.434000000000005
    - type: recall_at_100
      value: 76.632
    - type: recall_at_1000
      value: 90.25
    - type: recall_at_3
      value: 39.275
    - type: recall_at_5
      value: 46.225
    - type: map_at_1
      value: 17.885
    - type: map_at_10
      value: 25.724000000000004
    - type: map_at_100
      value: 26.992
    - type: map_at_1000
      value: 27.107999999999997
    - type: map_at_3
      value: 23.04
    - type: map_at_5
      value: 24.529
    - type: mrr_at_1
      value: 22.264
    - type: mrr_at_10
      value: 30.548
    - type: mrr_at_100
      value: 31.593
    - type: mrr_at_1000
      value: 31.657999999999998
    - type: mrr_at_3
      value: 27.756999999999998
    - type: mrr_at_5
      value: 29.398999999999997
    - type: ndcg_at_1
      value: 22.264
    - type: ndcg_at_10
      value: 30.902
    - type: ndcg_at_100
      value: 36.918
    - type: ndcg_at_1000
      value: 39.735
    - type: ndcg_at_3
      value: 25.915
    - type: ndcg_at_5
      value: 28.255999999999997
    - type: precision_at_1
      value: 22.264
    - type: precision_at_10
      value: 5.634
    - type: precision_at_100
      value: 0.9939999999999999
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 12.396
    - type: precision_at_5
      value: 9.055
    - type: recall_at_1
      value: 17.885
    - type: recall_at_10
      value: 42.237
    - type: recall_at_100
      value: 68.489
    - type: recall_at_1000
      value: 88.721
    - type: recall_at_3
      value: 28.283
    - type: recall_at_5
      value: 34.300000000000004
    - type: map_at_1
      value: 29.737000000000002
    - type: map_at_10
      value: 39.757
    - type: map_at_100
      value: 40.992
    - type: map_at_1000
      value: 41.102
    - type: map_at_3
      value: 36.612
    - type: map_at_5
      value: 38.413000000000004
    - type: mrr_at_1
      value: 35.804
    - type: mrr_at_10
      value: 45.178000000000004
    - type: mrr_at_100
      value: 45.975
    - type: mrr_at_1000
      value: 46.021
    - type: mrr_at_3
      value: 42.541000000000004
    - type: mrr_at_5
      value: 44.167
    - type: ndcg_at_1
      value: 35.804
    - type: ndcg_at_10
      value: 45.608
    - type: ndcg_at_100
      value: 50.746
    - type: ndcg_at_1000
      value: 52.839999999999996
    - type: ndcg_at_3
      value: 40.52
    - type: ndcg_at_5
      value: 43.051
    - type: precision_at_1
      value: 35.804
    - type: precision_at_10
      value: 8.104
    - type: precision_at_100
      value: 1.256
    - type: precision_at_1000
      value: 0.161
    - type: precision_at_3
      value: 19.121
    - type: precision_at_5
      value: 13.532
    - type: recall_at_1
      value: 29.737000000000002
    - type: recall_at_10
      value: 57.66
    - type: recall_at_100
      value: 79.121
    - type: recall_at_1000
      value: 93.023
    - type: recall_at_3
      value: 43.13
    - type: recall_at_5
      value: 49.836000000000006
    - type: map_at_1
      value: 26.299
    - type: map_at_10
      value: 35.617
    - type: map_at_100
      value: 36.972
    - type: map_at_1000
      value: 37.096000000000004
    - type: map_at_3
      value: 32.653999999999996
    - type: map_at_5
      value: 34.363
    - type: mrr_at_1
      value: 32.877
    - type: mrr_at_10
      value: 41.423
    - type: mrr_at_100
      value: 42.333999999999996
    - type: mrr_at_1000
      value: 42.398
    - type: mrr_at_3
      value: 39.193
    - type: mrr_at_5
      value: 40.426
    - type: ndcg_at_1
      value: 32.877
    - type: ndcg_at_10
      value: 41.271
    - type: ndcg_at_100
      value: 46.843
    - type: ndcg_at_1000
      value: 49.366
    - type: ndcg_at_3
      value: 36.735
    - type: ndcg_at_5
      value: 38.775999999999996
    - type: precision_at_1
      value: 32.877
    - type: precision_at_10
      value: 7.580000000000001
    - type: precision_at_100
      value: 1.192
    - type: precision_at_1000
      value: 0.158
    - type: precision_at_3
      value: 17.541999999999998
    - type: precision_at_5
      value: 12.443
    - type: recall_at_1
      value: 26.299
    - type: recall_at_10
      value: 52.256
    - type: recall_at_100
      value: 75.919
    - type: recall_at_1000
      value: 93.185
    - type: recall_at_3
      value: 39.271
    - type: recall_at_5
      value: 44.901
    - type: map_at_1
      value: 27.05741666666667
    - type: map_at_10
      value: 36.086416666666665
    - type: map_at_100
      value: 37.26916666666667
    - type: map_at_1000
      value: 37.38191666666666
    - type: map_at_3
      value: 33.34225
    - type: map_at_5
      value: 34.86425
    - type: mrr_at_1
      value: 32.06008333333333
    - type: mrr_at_10
      value: 40.36658333333333
    - type: mrr_at_100
      value: 41.206500000000005
    - type: mrr_at_1000
      value: 41.261083333333325
    - type: mrr_at_3
      value: 38.01208333333334
    - type: mrr_at_5
      value: 39.36858333333333
    - type: ndcg_at_1
      value: 32.06008333333333
    - type: ndcg_at_10
      value: 41.3535
    - type: ndcg_at_100
      value: 46.42066666666666
    - type: ndcg_at_1000
      value: 48.655166666666666
    - type: ndcg_at_3
      value: 36.78041666666667
    - type: ndcg_at_5
      value: 38.91783333333334
    - type: precision_at_1
      value: 32.06008333333333
    - type: precision_at_10
      value: 7.169833333333332
    - type: precision_at_100
      value: 1.1395
    - type: precision_at_1000
      value: 0.15158333333333332
    - type: precision_at_3
      value: 16.852
    - type: precision_at_5
      value: 11.8645
    - type: recall_at_1
      value: 27.05741666666667
    - type: recall_at_10
      value: 52.64491666666666
    - type: recall_at_100
      value: 74.99791666666667
    - type: recall_at_1000
      value: 90.50524999999999
    - type: recall_at_3
      value: 39.684000000000005
    - type: recall_at_5
      value: 45.37225
    - type: map_at_1
      value: 25.607999999999997
    - type: map_at_10
      value: 32.28
    - type: map_at_100
      value: 33.261
    - type: map_at_1000
      value: 33.346
    - type: map_at_3
      value: 30.514999999999997
    - type: map_at_5
      value: 31.415
    - type: mrr_at_1
      value: 28.988000000000003
    - type: mrr_at_10
      value: 35.384
    - type: mrr_at_100
      value: 36.24
    - type: mrr_at_1000
      value: 36.299
    - type: mrr_at_3
      value: 33.717000000000006
    - type: mrr_at_5
      value: 34.507
    - type: ndcg_at_1
      value: 28.988000000000003
    - type: ndcg_at_10
      value: 36.248000000000005
    - type: ndcg_at_100
      value: 41.034
    - type: ndcg_at_1000
      value: 43.35
    - type: ndcg_at_3
      value: 32.987
    - type: ndcg_at_5
      value: 34.333999999999996
    - type: precision_at_1
      value: 28.988000000000003
    - type: precision_at_10
      value: 5.506
    - type: precision_at_100
      value: 0.853
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 14.11
    - type: precision_at_5
      value: 9.417
    - type: recall_at_1
      value: 25.607999999999997
    - type: recall_at_10
      value: 45.344
    - type: recall_at_100
      value: 67.132
    - type: recall_at_1000
      value: 84.676
    - type: recall_at_3
      value: 36.02
    - type: recall_at_5
      value: 39.613
    - type: map_at_1
      value: 18.44
    - type: map_at_10
      value: 25.651000000000003
    - type: map_at_100
      value: 26.735
    - type: map_at_1000
      value: 26.86
    - type: map_at_3
      value: 23.409
    - type: map_at_5
      value: 24.604
    - type: mrr_at_1
      value: 22.195
    - type: mrr_at_10
      value: 29.482000000000003
    - type: mrr_at_100
      value: 30.395
    - type: mrr_at_1000
      value: 30.471999999999998
    - type: mrr_at_3
      value: 27.409
    - type: mrr_at_5
      value: 28.553
    - type: ndcg_at_1
      value: 22.195
    - type: ndcg_at_10
      value: 30.242
    - type: ndcg_at_100
      value: 35.397
    - type: ndcg_at_1000
      value: 38.287
    - type: ndcg_at_3
      value: 26.201
    - type: ndcg_at_5
      value: 28.008
    - type: precision_at_1
      value: 22.195
    - type: precision_at_10
      value: 5.372
    - type: precision_at_100
      value: 0.9259999999999999
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 12.228
    - type: precision_at_5
      value: 8.727
    - type: recall_at_1
      value: 18.44
    - type: recall_at_10
      value: 40.325
    - type: recall_at_100
      value: 63.504000000000005
    - type: recall_at_1000
      value: 83.909
    - type: recall_at_3
      value: 28.925
    - type: recall_at_5
      value: 33.641
    - type: map_at_1
      value: 26.535999999999998
    - type: map_at_10
      value: 35.358000000000004
    - type: map_at_100
      value: 36.498999999999995
    - type: map_at_1000
      value: 36.597
    - type: map_at_3
      value: 32.598
    - type: map_at_5
      value: 34.185
    - type: mrr_at_1
      value: 31.25
    - type: mrr_at_10
      value: 39.593
    - type: mrr_at_100
      value: 40.443
    - type: mrr_at_1000
      value: 40.498
    - type: mrr_at_3
      value: 37.018
    - type: mrr_at_5
      value: 38.492
    - type: ndcg_at_1
      value: 31.25
    - type: ndcg_at_10
      value: 40.71
    - type: ndcg_at_100
      value: 46.079
    - type: ndcg_at_1000
      value: 48.287
    - type: ndcg_at_3
      value: 35.667
    - type: ndcg_at_5
      value: 38.080000000000005
    - type: precision_at_1
      value: 31.25
    - type: precision_at_10
      value: 6.847
    - type: precision_at_100
      value: 1.079
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 16.262
    - type: precision_at_5
      value: 11.455
    - type: recall_at_1
      value: 26.535999999999998
    - type: recall_at_10
      value: 52.92099999999999
    - type: recall_at_100
      value: 76.669
    - type: recall_at_1000
      value: 92.096
    - type: recall_at_3
      value: 38.956
    - type: recall_at_5
      value: 45.239000000000004
    - type: map_at_1
      value: 24.691
    - type: map_at_10
      value: 33.417
    - type: map_at_100
      value: 35.036
    - type: map_at_1000
      value: 35.251
    - type: map_at_3
      value: 30.646
    - type: map_at_5
      value: 32.177
    - type: mrr_at_1
      value: 30.04
    - type: mrr_at_10
      value: 37.905
    - type: mrr_at_100
      value: 38.929
    - type: mrr_at_1000
      value: 38.983000000000004
    - type: mrr_at_3
      value: 35.276999999999994
    - type: mrr_at_5
      value: 36.897000000000006
    - type: ndcg_at_1
      value: 30.04
    - type: ndcg_at_10
      value: 39.037
    - type: ndcg_at_100
      value: 44.944
    - type: ndcg_at_1000
      value: 47.644
    - type: ndcg_at_3
      value: 34.833999999999996
    - type: ndcg_at_5
      value: 36.83
    - type: precision_at_1
      value: 30.04
    - type: precision_at_10
      value: 7.4510000000000005
    - type: precision_at_100
      value: 1.492
    - type: precision_at_1000
      value: 0.234
    - type: precision_at_3
      value: 16.337
    - type: precision_at_5
      value: 11.897
    - type: recall_at_1
      value: 24.691
    - type: recall_at_10
      value: 49.303999999999995
    - type: recall_at_100
      value: 76.20400000000001
    - type: recall_at_1000
      value: 93.30000000000001
    - type: recall_at_3
      value: 36.594
    - type: recall_at_5
      value: 42.41
    - type: map_at_1
      value: 23.118
    - type: map_at_10
      value: 30.714999999999996
    - type: map_at_100
      value: 31.656000000000002
    - type: map_at_1000
      value: 31.757
    - type: map_at_3
      value: 28.355000000000004
    - type: map_at_5
      value: 29.337000000000003
    - type: mrr_at_1
      value: 25.323
    - type: mrr_at_10
      value: 32.93
    - type: mrr_at_100
      value: 33.762
    - type: mrr_at_1000
      value: 33.829
    - type: mrr_at_3
      value: 30.775999999999996
    - type: mrr_at_5
      value: 31.774
    - type: ndcg_at_1
      value: 25.323
    - type: ndcg_at_10
      value: 35.408
    - type: ndcg_at_100
      value: 40.083
    - type: ndcg_at_1000
      value: 42.542
    - type: ndcg_at_3
      value: 30.717
    - type: ndcg_at_5
      value: 32.385000000000005
    - type: precision_at_1
      value: 25.323
    - type: precision_at_10
      value: 5.564
    - type: precision_at_100
      value: 0.843
    - type: precision_at_1000
      value: 0.116
    - type: precision_at_3
      value: 13.001
    - type: precision_at_5
      value: 8.834999999999999
    - type: recall_at_1
      value: 23.118
    - type: recall_at_10
      value: 47.788000000000004
    - type: recall_at_100
      value: 69.37
    - type: recall_at_1000
      value: 87.47399999999999
    - type: recall_at_3
      value: 34.868
    - type: recall_at_5
      value: 39.001999999999995
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
      value: 14.288
    - type: map_at_10
      value: 23.256
    - type: map_at_100
      value: 25.115
    - type: map_at_1000
      value: 25.319000000000003
    - type: map_at_3
      value: 20.005
    - type: map_at_5
      value: 21.529999999999998
    - type: mrr_at_1
      value: 31.401
    - type: mrr_at_10
      value: 42.251
    - type: mrr_at_100
      value: 43.236999999999995
    - type: mrr_at_1000
      value: 43.272
    - type: mrr_at_3
      value: 39.164
    - type: mrr_at_5
      value: 40.881
    - type: ndcg_at_1
      value: 31.401
    - type: ndcg_at_10
      value: 31.615
    - type: ndcg_at_100
      value: 38.982
    - type: ndcg_at_1000
      value: 42.496
    - type: ndcg_at_3
      value: 26.608999999999998
    - type: ndcg_at_5
      value: 28.048000000000002
    - type: precision_at_1
      value: 31.401
    - type: precision_at_10
      value: 9.536999999999999
    - type: precision_at_100
      value: 1.763
    - type: precision_at_1000
      value: 0.241
    - type: precision_at_3
      value: 19.153000000000002
    - type: precision_at_5
      value: 14.228
    - type: recall_at_1
      value: 14.288
    - type: recall_at_10
      value: 36.717
    - type: recall_at_100
      value: 61.9
    - type: recall_at_1000
      value: 81.676
    - type: recall_at_3
      value: 24.203
    - type: recall_at_5
      value: 28.793999999999997
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
      value: 9.019
    - type: map_at_10
      value: 19.963
    - type: map_at_100
      value: 28.834
    - type: map_at_1000
      value: 30.537999999999997
    - type: map_at_3
      value: 14.45
    - type: map_at_5
      value: 16.817999999999998
    - type: mrr_at_1
      value: 65.75
    - type: mrr_at_10
      value: 74.646
    - type: mrr_at_100
      value: 74.946
    - type: mrr_at_1000
      value: 74.95100000000001
    - type: mrr_at_3
      value: 72.625
    - type: mrr_at_5
      value: 74.012
    - type: ndcg_at_1
      value: 54
    - type: ndcg_at_10
      value: 42.014
    - type: ndcg_at_100
      value: 47.527
    - type: ndcg_at_1000
      value: 54.911
    - type: ndcg_at_3
      value: 46.586
    - type: ndcg_at_5
      value: 43.836999999999996
    - type: precision_at_1
      value: 65.75
    - type: precision_at_10
      value: 33.475
    - type: precision_at_100
      value: 11.16
    - type: precision_at_1000
      value: 2.145
    - type: precision_at_3
      value: 50.083
    - type: precision_at_5
      value: 42.55
    - type: recall_at_1
      value: 9.019
    - type: recall_at_10
      value: 25.558999999999997
    - type: recall_at_100
      value: 53.937999999999995
    - type: recall_at_1000
      value: 77.67399999999999
    - type: recall_at_3
      value: 15.456
    - type: recall_at_5
      value: 19.259
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
      value: 52.635
    - type: f1
      value: 47.692783881403926
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
      value: 76.893
    - type: map_at_10
      value: 84.897
    - type: map_at_100
      value: 85.122
    - type: map_at_1000
      value: 85.135
    - type: map_at_3
      value: 83.88
    - type: map_at_5
      value: 84.565
    - type: mrr_at_1
      value: 83.003
    - type: mrr_at_10
      value: 89.506
    - type: mrr_at_100
      value: 89.574
    - type: mrr_at_1000
      value: 89.575
    - type: mrr_at_3
      value: 88.991
    - type: mrr_at_5
      value: 89.349
    - type: ndcg_at_1
      value: 83.003
    - type: ndcg_at_10
      value: 88.351
    - type: ndcg_at_100
      value: 89.128
    - type: ndcg_at_1000
      value: 89.34100000000001
    - type: ndcg_at_3
      value: 86.92
    - type: ndcg_at_5
      value: 87.78200000000001
    - type: precision_at_1
      value: 83.003
    - type: precision_at_10
      value: 10.517999999999999
    - type: precision_at_100
      value: 1.115
    - type: precision_at_1000
      value: 0.11499999999999999
    - type: precision_at_3
      value: 33.062999999999995
    - type: precision_at_5
      value: 20.498
    - type: recall_at_1
      value: 76.893
    - type: recall_at_10
      value: 94.374
    - type: recall_at_100
      value: 97.409
    - type: recall_at_1000
      value: 98.687
    - type: recall_at_3
      value: 90.513
    - type: recall_at_5
      value: 92.709
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
      value: 20.829
    - type: map_at_10
      value: 32.86
    - type: map_at_100
      value: 34.838
    - type: map_at_1000
      value: 35.006
    - type: map_at_3
      value: 28.597
    - type: map_at_5
      value: 31.056
    - type: mrr_at_1
      value: 41.358
    - type: mrr_at_10
      value: 49.542
    - type: mrr_at_100
      value: 50.29900000000001
    - type: mrr_at_1000
      value: 50.334999999999994
    - type: mrr_at_3
      value: 46.579
    - type: mrr_at_5
      value: 48.408
    - type: ndcg_at_1
      value: 41.358
    - type: ndcg_at_10
      value: 40.758
    - type: ndcg_at_100
      value: 47.799
    - type: ndcg_at_1000
      value: 50.589
    - type: ndcg_at_3
      value: 36.695
    - type: ndcg_at_5
      value: 38.193
    - type: precision_at_1
      value: 41.358
    - type: precision_at_10
      value: 11.142000000000001
    - type: precision_at_100
      value: 1.8350000000000002
    - type: precision_at_1000
      value: 0.234
    - type: precision_at_3
      value: 24.023
    - type: precision_at_5
      value: 17.963
    - type: recall_at_1
      value: 20.829
    - type: recall_at_10
      value: 47.467999999999996
    - type: recall_at_100
      value: 73.593
    - type: recall_at_1000
      value: 90.122
    - type: recall_at_3
      value: 32.74
    - type: recall_at_5
      value: 39.608
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
      value: 40.324
    - type: map_at_10
      value: 64.183
    - type: map_at_100
      value: 65.037
    - type: map_at_1000
      value: 65.094
    - type: map_at_3
      value: 60.663
    - type: map_at_5
      value: 62.951
    - type: mrr_at_1
      value: 80.648
    - type: mrr_at_10
      value: 86.005
    - type: mrr_at_100
      value: 86.157
    - type: mrr_at_1000
      value: 86.162
    - type: mrr_at_3
      value: 85.116
    - type: mrr_at_5
      value: 85.703
    - type: ndcg_at_1
      value: 80.648
    - type: ndcg_at_10
      value: 72.351
    - type: ndcg_at_100
      value: 75.279
    - type: ndcg_at_1000
      value: 76.357
    - type: ndcg_at_3
      value: 67.484
    - type: ndcg_at_5
      value: 70.31500000000001
    - type: precision_at_1
      value: 80.648
    - type: precision_at_10
      value: 15.103
    - type: precision_at_100
      value: 1.7399999999999998
    - type: precision_at_1000
      value: 0.188
    - type: precision_at_3
      value: 43.232
    - type: precision_at_5
      value: 28.165000000000003
    - type: recall_at_1
      value: 40.324
    - type: recall_at_10
      value: 75.517
    - type: recall_at_100
      value: 86.982
    - type: recall_at_1000
      value: 94.072
    - type: recall_at_3
      value: 64.848
    - type: recall_at_5
      value: 70.41199999999999
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
      value: 91.4
    - type: ap
      value: 87.4422032289312
    - type: f1
      value: 91.39249564302281
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
      value: 22.03
    - type: map_at_10
      value: 34.402
    - type: map_at_100
      value: 35.599
    - type: map_at_1000
      value: 35.648
    - type: map_at_3
      value: 30.603
    - type: map_at_5
      value: 32.889
    - type: mrr_at_1
      value: 22.679
    - type: mrr_at_10
      value: 35.021
    - type: mrr_at_100
      value: 36.162
    - type: mrr_at_1000
      value: 36.205
    - type: mrr_at_3
      value: 31.319999999999997
    - type: mrr_at_5
      value: 33.562
    - type: ndcg_at_1
      value: 22.692999999999998
    - type: ndcg_at_10
      value: 41.258
    - type: ndcg_at_100
      value: 46.967
    - type: ndcg_at_1000
      value: 48.175000000000004
    - type: ndcg_at_3
      value: 33.611000000000004
    - type: ndcg_at_5
      value: 37.675
    - type: precision_at_1
      value: 22.692999999999998
    - type: precision_at_10
      value: 6.5089999999999995
    - type: precision_at_100
      value: 0.936
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.413
    - type: precision_at_5
      value: 10.702
    - type: recall_at_1
      value: 22.03
    - type: recall_at_10
      value: 62.248000000000005
    - type: recall_at_100
      value: 88.524
    - type: recall_at_1000
      value: 97.714
    - type: recall_at_3
      value: 41.617
    - type: recall_at_5
      value: 51.359
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
      value: 94.36844505243957
    - type: f1
      value: 94.12408743818202
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
      value: 76.43410852713177
    - type: f1
      value: 58.501855709435624
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
      value: 76.04909213180902
    - type: f1
      value: 74.1800860395823
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
      value: 79.76126429051781
    - type: f1
      value: 79.85705217473232
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
      value: 34.70119520292863
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
      value: 32.33544316467486
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
      value: 30.75499243990726
    - type: mrr
      value: 31.70602251821063
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
      value: 6.451999999999999
    - type: map_at_10
      value: 13.918
    - type: map_at_100
      value: 17.316000000000003
    - type: map_at_1000
      value: 18.747
    - type: map_at_3
      value: 10.471
    - type: map_at_5
      value: 12.104
    - type: mrr_at_1
      value: 46.749
    - type: mrr_at_10
      value: 55.717000000000006
    - type: mrr_at_100
      value: 56.249
    - type: mrr_at_1000
      value: 56.288000000000004
    - type: mrr_at_3
      value: 53.818
    - type: mrr_at_5
      value: 55.103
    - type: ndcg_at_1
      value: 45.201
    - type: ndcg_at_10
      value: 35.539
    - type: ndcg_at_100
      value: 32.586
    - type: ndcg_at_1000
      value: 41.486000000000004
    - type: ndcg_at_3
      value: 41.174
    - type: ndcg_at_5
      value: 38.939
    - type: precision_at_1
      value: 46.749
    - type: precision_at_10
      value: 25.944
    - type: precision_at_100
      value: 8.084
    - type: precision_at_1000
      value: 2.076
    - type: precision_at_3
      value: 38.7
    - type: precision_at_5
      value: 33.56
    - type: recall_at_1
      value: 6.451999999999999
    - type: recall_at_10
      value: 17.302
    - type: recall_at_100
      value: 32.14
    - type: recall_at_1000
      value: 64.12
    - type: recall_at_3
      value: 11.219
    - type: recall_at_5
      value: 13.993
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
      value: 32.037
    - type: map_at_10
      value: 46.565
    - type: map_at_100
      value: 47.606
    - type: map_at_1000
      value: 47.636
    - type: map_at_3
      value: 42.459
    - type: map_at_5
      value: 44.762
    - type: mrr_at_1
      value: 36.181999999999995
    - type: mrr_at_10
      value: 49.291000000000004
    - type: mrr_at_100
      value: 50.059
    - type: mrr_at_1000
      value: 50.078
    - type: mrr_at_3
      value: 45.829
    - type: mrr_at_5
      value: 47.797
    - type: ndcg_at_1
      value: 36.153
    - type: ndcg_at_10
      value: 53.983000000000004
    - type: ndcg_at_100
      value: 58.347
    - type: ndcg_at_1000
      value: 59.058
    - type: ndcg_at_3
      value: 46.198
    - type: ndcg_at_5
      value: 50.022
    - type: precision_at_1
      value: 36.153
    - type: precision_at_10
      value: 8.763
    - type: precision_at_100
      value: 1.123
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 20.751
    - type: precision_at_5
      value: 14.646999999999998
    - type: recall_at_1
      value: 32.037
    - type: recall_at_10
      value: 74.008
    - type: recall_at_100
      value: 92.893
    - type: recall_at_1000
      value: 98.16
    - type: recall_at_3
      value: 53.705999999999996
    - type: recall_at_5
      value: 62.495
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
      value: 71.152
    - type: map_at_10
      value: 85.104
    - type: map_at_100
      value: 85.745
    - type: map_at_1000
      value: 85.761
    - type: map_at_3
      value: 82.175
    - type: map_at_5
      value: 84.066
    - type: mrr_at_1
      value: 82.03
    - type: mrr_at_10
      value: 88.115
    - type: mrr_at_100
      value: 88.21
    - type: mrr_at_1000
      value: 88.211
    - type: mrr_at_3
      value: 87.19200000000001
    - type: mrr_at_5
      value: 87.85
    - type: ndcg_at_1
      value: 82.03
    - type: ndcg_at_10
      value: 88.78
    - type: ndcg_at_100
      value: 89.96300000000001
    - type: ndcg_at_1000
      value: 90.056
    - type: ndcg_at_3
      value: 86.051
    - type: ndcg_at_5
      value: 87.63499999999999
    - type: precision_at_1
      value: 82.03
    - type: precision_at_10
      value: 13.450000000000001
    - type: precision_at_100
      value: 1.5310000000000001
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.627
    - type: precision_at_5
      value: 24.784
    - type: recall_at_1
      value: 71.152
    - type: recall_at_10
      value: 95.649
    - type: recall_at_100
      value: 99.58200000000001
    - type: recall_at_1000
      value: 99.981
    - type: recall_at_3
      value: 87.767
    - type: recall_at_5
      value: 92.233
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
      value: 56.48713646277477
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
      value: 63.394940772438545
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
      value: 5.043
    - type: map_at_10
      value: 12.949
    - type: map_at_100
      value: 15.146
    - type: map_at_1000
      value: 15.495000000000001
    - type: map_at_3
      value: 9.333
    - type: map_at_5
      value: 11.312999999999999
    - type: mrr_at_1
      value: 24.9
    - type: mrr_at_10
      value: 35.958
    - type: mrr_at_100
      value: 37.152
    - type: mrr_at_1000
      value: 37.201
    - type: mrr_at_3
      value: 32.667
    - type: mrr_at_5
      value: 34.567
    - type: ndcg_at_1
      value: 24.9
    - type: ndcg_at_10
      value: 21.298000000000002
    - type: ndcg_at_100
      value: 29.849999999999998
    - type: ndcg_at_1000
      value: 35.506
    - type: ndcg_at_3
      value: 20.548
    - type: ndcg_at_5
      value: 18.064
    - type: precision_at_1
      value: 24.9
    - type: precision_at_10
      value: 10.9
    - type: precision_at_100
      value: 2.331
    - type: precision_at_1000
      value: 0.367
    - type: precision_at_3
      value: 19.267
    - type: precision_at_5
      value: 15.939999999999998
    - type: recall_at_1
      value: 5.043
    - type: recall_at_10
      value: 22.092
    - type: recall_at_100
      value: 47.323
    - type: recall_at_1000
      value: 74.553
    - type: recall_at_3
      value: 11.728
    - type: recall_at_5
      value: 16.188
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
      value: 83.7007085938325
    - type: cos_sim_spearman
      value: 80.0171084446234
    - type: euclidean_pearson
      value: 81.28133218355893
    - type: euclidean_spearman
      value: 79.99291731740131
    - type: manhattan_pearson
      value: 81.22926922327846
    - type: manhattan_spearman
      value: 79.94444878127038
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
      value: 85.7411883252923
    - type: cos_sim_spearman
      value: 77.93462937801245
    - type: euclidean_pearson
      value: 83.00858563882404
    - type: euclidean_spearman
      value: 77.82717362433257
    - type: manhattan_pearson
      value: 82.92887645790769
    - type: manhattan_spearman
      value: 77.78807488222115
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
      value: 82.04222459361023
    - type: cos_sim_spearman
      value: 83.85931509330395
    - type: euclidean_pearson
      value: 83.26916063876055
    - type: euclidean_spearman
      value: 83.98621985648353
    - type: manhattan_pearson
      value: 83.14935679184327
    - type: manhattan_spearman
      value: 83.87938828586304
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
      value: 81.41136639535318
    - type: cos_sim_spearman
      value: 81.51200091040481
    - type: euclidean_pearson
      value: 81.45382456114775
    - type: euclidean_spearman
      value: 81.46201181707931
    - type: manhattan_pearson
      value: 81.37243088439584
    - type: manhattan_spearman
      value: 81.39828421893426
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
      value: 85.71942451732227
    - type: cos_sim_spearman
      value: 87.33044482064973
    - type: euclidean_pearson
      value: 86.58580899365178
    - type: euclidean_spearman
      value: 87.09206723832895
    - type: manhattan_pearson
      value: 86.47460784157013
    - type: manhattan_spearman
      value: 86.98367656583076
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
      value: 83.55868078863449
    - type: cos_sim_spearman
      value: 85.38299230074065
    - type: euclidean_pearson
      value: 84.64715256244595
    - type: euclidean_spearman
      value: 85.49112229604047
    - type: manhattan_pearson
      value: 84.60814346792462
    - type: manhattan_spearman
      value: 85.44886026766822
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
      value: 84.99292526370614
    - type: cos_sim_spearman
      value: 85.58139465695983
    - type: euclidean_pearson
      value: 86.51325066734084
    - type: euclidean_spearman
      value: 85.56736418284562
    - type: manhattan_pearson
      value: 86.48190836601357
    - type: manhattan_spearman
      value: 85.51616256224258
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
      value: 64.54124715078807
    - type: cos_sim_spearman
      value: 65.32134275948374
    - type: euclidean_pearson
      value: 67.09791698300816
    - type: euclidean_spearman
      value: 65.79468982468465
    - type: manhattan_pearson
      value: 67.13304723693966
    - type: manhattan_spearman
      value: 65.68439995849283
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
      value: 83.4231099581624
    - type: cos_sim_spearman
      value: 85.95475815226862
    - type: euclidean_pearson
      value: 85.00339401999706
    - type: euclidean_spearman
      value: 85.74133081802971
    - type: manhattan_pearson
      value: 85.00407987181666
    - type: manhattan_spearman
      value: 85.77509596397363
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
      value: 87.25666719585716
    - type: mrr
      value: 96.32769917083642
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
      value: 57.828
    - type: map_at_10
      value: 68.369
    - type: map_at_100
      value: 68.83399999999999
    - type: map_at_1000
      value: 68.856
    - type: map_at_3
      value: 65.38000000000001
    - type: map_at_5
      value: 67.06299999999999
    - type: mrr_at_1
      value: 61
    - type: mrr_at_10
      value: 69.45400000000001
    - type: mrr_at_100
      value: 69.785
    - type: mrr_at_1000
      value: 69.807
    - type: mrr_at_3
      value: 67
    - type: mrr_at_5
      value: 68.43299999999999
    - type: ndcg_at_1
      value: 61
    - type: ndcg_at_10
      value: 73.258
    - type: ndcg_at_100
      value: 75.173
    - type: ndcg_at_1000
      value: 75.696
    - type: ndcg_at_3
      value: 68.162
    - type: ndcg_at_5
      value: 70.53399999999999
    - type: precision_at_1
      value: 61
    - type: precision_at_10
      value: 9.8
    - type: precision_at_100
      value: 1.087
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 27
    - type: precision_at_5
      value: 17.666999999999998
    - type: recall_at_1
      value: 57.828
    - type: recall_at_10
      value: 87.122
    - type: recall_at_100
      value: 95.667
    - type: recall_at_1000
      value: 99.667
    - type: recall_at_3
      value: 73.139
    - type: recall_at_5
      value: 79.361
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
      value: 99.85247524752475
    - type: cos_sim_ap
      value: 96.25640197639723
    - type: cos_sim_f1
      value: 92.37851662404091
    - type: cos_sim_precision
      value: 94.55497382198953
    - type: cos_sim_recall
      value: 90.3
    - type: dot_accuracy
      value: 99.76138613861386
    - type: dot_ap
      value: 93.40295864389073
    - type: dot_f1
      value: 87.64267990074441
    - type: dot_precision
      value: 86.99507389162562
    - type: dot_recall
      value: 88.3
    - type: euclidean_accuracy
      value: 99.85049504950496
    - type: euclidean_ap
      value: 96.24254350525462
    - type: euclidean_f1
      value: 92.32323232323232
    - type: euclidean_precision
      value: 93.26530612244898
    - type: euclidean_recall
      value: 91.4
    - type: manhattan_accuracy
      value: 99.85346534653465
    - type: manhattan_ap
      value: 96.2635334753325
    - type: manhattan_f1
      value: 92.37899073120495
    - type: manhattan_precision
      value: 95.22292993630573
    - type: manhattan_recall
      value: 89.7
    - type: max_accuracy
      value: 99.85346534653465
    - type: max_ap
      value: 96.2635334753325
    - type: max_f1
      value: 92.37899073120495
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
      value: 65.83905786483794
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
      value: 35.031896152126436
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
      value: 54.551326709447146
    - type: mrr
      value: 55.43758222986165
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
      value: 30.305688567308874
    - type: cos_sim_spearman
      value: 29.27135743434515
    - type: dot_pearson
      value: 30.336741878796563
    - type: dot_spearman
      value: 30.513365725895937
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
      value: 0.245
    - type: map_at_10
      value: 1.92
    - type: map_at_100
      value: 10.519
    - type: map_at_1000
      value: 23.874000000000002
    - type: map_at_3
      value: 0.629
    - type: map_at_5
      value: 1.0290000000000001
    - type: mrr_at_1
      value: 88
    - type: mrr_at_10
      value: 93.5
    - type: mrr_at_100
      value: 93.5
    - type: mrr_at_1000
      value: 93.5
    - type: mrr_at_3
      value: 93
    - type: mrr_at_5
      value: 93.5
    - type: ndcg_at_1
      value: 84
    - type: ndcg_at_10
      value: 76.447
    - type: ndcg_at_100
      value: 56.516
    - type: ndcg_at_1000
      value: 48.583999999999996
    - type: ndcg_at_3
      value: 78.877
    - type: ndcg_at_5
      value: 79.174
    - type: precision_at_1
      value: 88
    - type: precision_at_10
      value: 80.60000000000001
    - type: precision_at_100
      value: 57.64
    - type: precision_at_1000
      value: 21.227999999999998
    - type: precision_at_3
      value: 82
    - type: precision_at_5
      value: 83.6
    - type: recall_at_1
      value: 0.245
    - type: recall_at_10
      value: 2.128
    - type: recall_at_100
      value: 13.767
    - type: recall_at_1000
      value: 44.958
    - type: recall_at_3
      value: 0.654
    - type: recall_at_5
      value: 1.111
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
      value: 2.5170000000000003
    - type: map_at_10
      value: 10.915
    - type: map_at_100
      value: 17.535
    - type: map_at_1000
      value: 19.042
    - type: map_at_3
      value: 5.689
    - type: map_at_5
      value: 7.837
    - type: mrr_at_1
      value: 34.694
    - type: mrr_at_10
      value: 49.547999999999995
    - type: mrr_at_100
      value: 50.653000000000006
    - type: mrr_at_1000
      value: 50.653000000000006
    - type: mrr_at_3
      value: 44.558
    - type: mrr_at_5
      value: 48.333
    - type: ndcg_at_1
      value: 32.653
    - type: ndcg_at_10
      value: 26.543
    - type: ndcg_at_100
      value: 38.946
    - type: ndcg_at_1000
      value: 49.406
    - type: ndcg_at_3
      value: 29.903000000000002
    - type: ndcg_at_5
      value: 29.231
    - type: precision_at_1
      value: 34.694
    - type: precision_at_10
      value: 23.265
    - type: precision_at_100
      value: 8.102
    - type: precision_at_1000
      value: 1.5
    - type: precision_at_3
      value: 31.293
    - type: precision_at_5
      value: 29.796
    - type: recall_at_1
      value: 2.5170000000000003
    - type: recall_at_10
      value: 16.88
    - type: recall_at_100
      value: 49.381
    - type: recall_at_1000
      value: 81.23899999999999
    - type: recall_at_3
      value: 6.965000000000001
    - type: recall_at_5
      value: 10.847999999999999
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
      value: 71.5942
    - type: ap
      value: 13.92074156956546
    - type: f1
      value: 54.671999698839066
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
      value: 59.39728353140916
    - type: f1
      value: 59.68980496759517
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
      value: 52.11181870104935
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
      value: 86.46957143708649
    - type: cos_sim_ap
      value: 76.16120197845457
    - type: cos_sim_f1
      value: 69.69919295671315
    - type: cos_sim_precision
      value: 64.94986326344576
    - type: cos_sim_recall
      value: 75.19788918205805
    - type: dot_accuracy
      value: 83.0780234845324
    - type: dot_ap
      value: 64.21717343541934
    - type: dot_f1
      value: 59.48375497624245
    - type: dot_precision
      value: 57.94345759319489
    - type: dot_recall
      value: 61.108179419525065
    - type: euclidean_accuracy
      value: 86.6543482148179
    - type: euclidean_ap
      value: 76.4527555010203
    - type: euclidean_f1
      value: 70.10156056477584
    - type: euclidean_precision
      value: 66.05975723622782
    - type: euclidean_recall
      value: 74.67018469656992
    - type: manhattan_accuracy
      value: 86.66030875603504
    - type: manhattan_ap
      value: 76.40304567255436
    - type: manhattan_f1
      value: 70.05275426328058
    - type: manhattan_precision
      value: 65.4666360926393
    - type: manhattan_recall
      value: 75.32981530343008
    - type: max_accuracy
      value: 86.66030875603504
    - type: max_ap
      value: 76.4527555010203
    - type: max_f1
      value: 70.10156056477584
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
      value: 88.42123646524624
    - type: cos_sim_ap
      value: 85.15431437761646
    - type: cos_sim_f1
      value: 76.98069301530742
    - type: cos_sim_precision
      value: 72.9314502239063
    - type: cos_sim_recall
      value: 81.50600554357868
    - type: dot_accuracy
      value: 86.70974502270346
    - type: dot_ap
      value: 80.77621563599457
    - type: dot_f1
      value: 73.87058697285117
    - type: dot_precision
      value: 68.98256396552877
    - type: dot_recall
      value: 79.50415768401602
    - type: euclidean_accuracy
      value: 88.46392672798541
    - type: euclidean_ap
      value: 85.20370297495491
    - type: euclidean_f1
      value: 77.01372369624886
    - type: euclidean_precision
      value: 73.39052800446397
    - type: euclidean_recall
      value: 81.01324299353249
    - type: manhattan_accuracy
      value: 88.43481973066325
    - type: manhattan_ap
      value: 85.16318289864545
    - type: manhattan_f1
      value: 76.90884877182597
    - type: manhattan_precision
      value: 74.01737396753062
    - type: manhattan_recall
      value: 80.03541730828458
    - type: max_accuracy
      value: 88.46392672798541
    - type: max_ap
      value: 85.20370297495491
    - type: max_f1
      value: 77.01372369624886
---


**Recommend switching to newest [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), which has more reasonable similarity distribution and same method of usage.**

<h1 align="center">FlagEmbedding</h1>


<h4 align="center">
    <p>
        <a href=#model-list>Model List</a> | 
        <a href=#frequently-asked-questions>FAQ</a> |
        <a href=#usage>Usage</a>  |
        <a href="#evaluation">Evaluation</a> |
        <a href="#train">Train</a> |
        <a href="#contact">Contact</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

More details please refer to our Github: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).


[English](README.md) | [中文](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

FlagEmbedding can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification,  clustering, or semantic search.
And it also can be used in vector databases for LLMs.

************* 🌟**Updates**🌟 *************
- 10/12/2023: Release [LLM-Embedder](./FlagEmbedding/llm_embedder/README.md), a unified embedding model to support diverse retrieval augmentation needs for LLMs. [Paper](https://arxiv.org/pdf/2310.07554.pdf)  :fire:  
- 09/15/2023: The [technical report](https://arxiv.org/pdf/2309.07597.pdf) of BGE has been released 
- 09/15/2023: The [masive training data](https://data.baai.ac.cn/details/BAAI-MTP) of BGE has been released 
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


## Contact
If you have any question or suggestion related to this project, feel free to open an issue or pull request.
You also can email Shitao Xiao(stxiao@baai.ac.cn) and Zheng Liu(liuzheng@baai.ac.cn). 


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

