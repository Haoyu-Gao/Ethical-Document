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
- name: e5-large
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
      value: 77.68656716417911
    - type: ap
      value: 41.336896075573584
    - type: f1
      value: 71.788561468075
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
      value: 90.04965
    - type: ap
      value: 86.24637009569418
    - type: f1
      value: 90.03896671762645
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
      value: 43.016000000000005
    - type: f1
      value: 42.1942431880186
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
      value: 25.107000000000003
    - type: map_at_10
      value: 40.464
    - type: map_at_100
      value: 41.577999999999996
    - type: map_at_1000
      value: 41.588
    - type: map_at_3
      value: 35.301
    - type: map_at_5
      value: 38.263000000000005
    - type: mrr_at_1
      value: 25.605
    - type: mrr_at_10
      value: 40.64
    - type: mrr_at_100
      value: 41.760000000000005
    - type: mrr_at_1000
      value: 41.77
    - type: mrr_at_3
      value: 35.443000000000005
    - type: mrr_at_5
      value: 38.448
    - type: ndcg_at_1
      value: 25.107000000000003
    - type: ndcg_at_10
      value: 49.352000000000004
    - type: ndcg_at_100
      value: 53.98500000000001
    - type: ndcg_at_1000
      value: 54.208
    - type: ndcg_at_3
      value: 38.671
    - type: ndcg_at_5
      value: 43.991
    - type: precision_at_1
      value: 25.107000000000003
    - type: precision_at_10
      value: 7.795000000000001
    - type: precision_at_100
      value: 0.979
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 16.145
    - type: precision_at_5
      value: 12.262
    - type: recall_at_1
      value: 25.107000000000003
    - type: recall_at_10
      value: 77.952
    - type: recall_at_100
      value: 97.866
    - type: recall_at_1000
      value: 99.57300000000001
    - type: recall_at_3
      value: 48.435
    - type: recall_at_5
      value: 61.309000000000005
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
      value: 46.19278045044154
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
      value: 41.37976387757665
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
      value: 60.07433334608074
    - type: mrr
      value: 73.44347711383723
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
      value: 86.4298072183543
    - type: cos_sim_spearman
      value: 84.73144873582848
    - type: euclidean_pearson
      value: 85.15885058870728
    - type: euclidean_spearman
      value: 85.42062106559356
    - type: manhattan_pearson
      value: 84.89409921792054
    - type: manhattan_spearman
      value: 85.31941394024344
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
      value: 84.14285714285714
    - type: f1
      value: 84.11674412565644
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
      value: 37.600076342340785
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
      value: 35.08861812135148
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
      value: 32.684000000000005
    - type: map_at_10
      value: 41.675000000000004
    - type: map_at_100
      value: 42.963
    - type: map_at_1000
      value: 43.078
    - type: map_at_3
      value: 38.708999999999996
    - type: map_at_5
      value: 40.316
    - type: mrr_at_1
      value: 39.485
    - type: mrr_at_10
      value: 47.152
    - type: mrr_at_100
      value: 47.96
    - type: mrr_at_1000
      value: 48.010000000000005
    - type: mrr_at_3
      value: 44.754
    - type: mrr_at_5
      value: 46.285
    - type: ndcg_at_1
      value: 39.485
    - type: ndcg_at_10
      value: 46.849000000000004
    - type: ndcg_at_100
      value: 52.059
    - type: ndcg_at_1000
      value: 54.358
    - type: ndcg_at_3
      value: 42.705
    - type: ndcg_at_5
      value: 44.663000000000004
    - type: precision_at_1
      value: 39.485
    - type: precision_at_10
      value: 8.455
    - type: precision_at_100
      value: 1.3379999999999999
    - type: precision_at_1000
      value: 0.178
    - type: precision_at_3
      value: 19.695
    - type: precision_at_5
      value: 13.905999999999999
    - type: recall_at_1
      value: 32.684000000000005
    - type: recall_at_10
      value: 56.227000000000004
    - type: recall_at_100
      value: 78.499
    - type: recall_at_1000
      value: 94.021
    - type: recall_at_3
      value: 44.157999999999994
    - type: recall_at_5
      value: 49.694
    - type: map_at_1
      value: 31.875999999999998
    - type: map_at_10
      value: 41.603
    - type: map_at_100
      value: 42.825
    - type: map_at_1000
      value: 42.961
    - type: map_at_3
      value: 38.655
    - type: map_at_5
      value: 40.294999999999995
    - type: mrr_at_1
      value: 40.127
    - type: mrr_at_10
      value: 47.959
    - type: mrr_at_100
      value: 48.59
    - type: mrr_at_1000
      value: 48.634
    - type: mrr_at_3
      value: 45.786
    - type: mrr_at_5
      value: 46.964
    - type: ndcg_at_1
      value: 40.127
    - type: ndcg_at_10
      value: 47.176
    - type: ndcg_at_100
      value: 51.346000000000004
    - type: ndcg_at_1000
      value: 53.502
    - type: ndcg_at_3
      value: 43.139
    - type: ndcg_at_5
      value: 44.883
    - type: precision_at_1
      value: 40.127
    - type: precision_at_10
      value: 8.72
    - type: precision_at_100
      value: 1.387
    - type: precision_at_1000
      value: 0.188
    - type: precision_at_3
      value: 20.637
    - type: precision_at_5
      value: 14.446
    - type: recall_at_1
      value: 31.875999999999998
    - type: recall_at_10
      value: 56.54900000000001
    - type: recall_at_100
      value: 73.939
    - type: recall_at_1000
      value: 87.732
    - type: recall_at_3
      value: 44.326
    - type: recall_at_5
      value: 49.445
    - type: map_at_1
      value: 41.677
    - type: map_at_10
      value: 52.222
    - type: map_at_100
      value: 53.229000000000006
    - type: map_at_1000
      value: 53.288000000000004
    - type: map_at_3
      value: 49.201
    - type: map_at_5
      value: 51.00599999999999
    - type: mrr_at_1
      value: 47.524
    - type: mrr_at_10
      value: 55.745999999999995
    - type: mrr_at_100
      value: 56.433
    - type: mrr_at_1000
      value: 56.464999999999996
    - type: mrr_at_3
      value: 53.37499999999999
    - type: mrr_at_5
      value: 54.858
    - type: ndcg_at_1
      value: 47.524
    - type: ndcg_at_10
      value: 57.406
    - type: ndcg_at_100
      value: 61.403
    - type: ndcg_at_1000
      value: 62.7
    - type: ndcg_at_3
      value: 52.298
    - type: ndcg_at_5
      value: 55.02
    - type: precision_at_1
      value: 47.524
    - type: precision_at_10
      value: 8.865
    - type: precision_at_100
      value: 1.179
    - type: precision_at_1000
      value: 0.134
    - type: precision_at_3
      value: 22.612
    - type: precision_at_5
      value: 15.461
    - type: recall_at_1
      value: 41.677
    - type: recall_at_10
      value: 69.346
    - type: recall_at_100
      value: 86.344
    - type: recall_at_1000
      value: 95.703
    - type: recall_at_3
      value: 55.789
    - type: recall_at_5
      value: 62.488
    - type: map_at_1
      value: 25.991999999999997
    - type: map_at_10
      value: 32.804
    - type: map_at_100
      value: 33.812999999999995
    - type: map_at_1000
      value: 33.897
    - type: map_at_3
      value: 30.567
    - type: map_at_5
      value: 31.599
    - type: mrr_at_1
      value: 27.797
    - type: mrr_at_10
      value: 34.768
    - type: mrr_at_100
      value: 35.702
    - type: mrr_at_1000
      value: 35.766
    - type: mrr_at_3
      value: 32.637
    - type: mrr_at_5
      value: 33.614
    - type: ndcg_at_1
      value: 27.797
    - type: ndcg_at_10
      value: 36.966
    - type: ndcg_at_100
      value: 41.972
    - type: ndcg_at_1000
      value: 44.139
    - type: ndcg_at_3
      value: 32.547
    - type: ndcg_at_5
      value: 34.258
    - type: precision_at_1
      value: 27.797
    - type: precision_at_10
      value: 5.514
    - type: precision_at_100
      value: 0.8340000000000001
    - type: precision_at_1000
      value: 0.106
    - type: precision_at_3
      value: 13.333
    - type: precision_at_5
      value: 9.04
    - type: recall_at_1
      value: 25.991999999999997
    - type: recall_at_10
      value: 47.941
    - type: recall_at_100
      value: 71.039
    - type: recall_at_1000
      value: 87.32799999999999
    - type: recall_at_3
      value: 36.01
    - type: recall_at_5
      value: 40.056000000000004
    - type: map_at_1
      value: 17.533
    - type: map_at_10
      value: 24.336
    - type: map_at_100
      value: 25.445
    - type: map_at_1000
      value: 25.561
    - type: map_at_3
      value: 22.116
    - type: map_at_5
      value: 23.347
    - type: mrr_at_1
      value: 21.642
    - type: mrr_at_10
      value: 28.910999999999998
    - type: mrr_at_100
      value: 29.836000000000002
    - type: mrr_at_1000
      value: 29.907
    - type: mrr_at_3
      value: 26.638
    - type: mrr_at_5
      value: 27.857
    - type: ndcg_at_1
      value: 21.642
    - type: ndcg_at_10
      value: 28.949
    - type: ndcg_at_100
      value: 34.211000000000006
    - type: ndcg_at_1000
      value: 37.031
    - type: ndcg_at_3
      value: 24.788
    - type: ndcg_at_5
      value: 26.685
    - type: precision_at_1
      value: 21.642
    - type: precision_at_10
      value: 5.137
    - type: precision_at_100
      value: 0.893
    - type: precision_at_1000
      value: 0.127
    - type: precision_at_3
      value: 11.733
    - type: precision_at_5
      value: 8.383000000000001
    - type: recall_at_1
      value: 17.533
    - type: recall_at_10
      value: 38.839
    - type: recall_at_100
      value: 61.458999999999996
    - type: recall_at_1000
      value: 81.58
    - type: recall_at_3
      value: 27.328999999999997
    - type: recall_at_5
      value: 32.168
    - type: map_at_1
      value: 28.126
    - type: map_at_10
      value: 37.872
    - type: map_at_100
      value: 39.229
    - type: map_at_1000
      value: 39.353
    - type: map_at_3
      value: 34.93
    - type: map_at_5
      value: 36.59
    - type: mrr_at_1
      value: 34.071
    - type: mrr_at_10
      value: 43.056
    - type: mrr_at_100
      value: 43.944
    - type: mrr_at_1000
      value: 43.999
    - type: mrr_at_3
      value: 40.536
    - type: mrr_at_5
      value: 42.065999999999995
    - type: ndcg_at_1
      value: 34.071
    - type: ndcg_at_10
      value: 43.503
    - type: ndcg_at_100
      value: 49.120000000000005
    - type: ndcg_at_1000
      value: 51.410999999999994
    - type: ndcg_at_3
      value: 38.767
    - type: ndcg_at_5
      value: 41.075
    - type: precision_at_1
      value: 34.071
    - type: precision_at_10
      value: 7.843999999999999
    - type: precision_at_100
      value: 1.2489999999999999
    - type: precision_at_1000
      value: 0.163
    - type: precision_at_3
      value: 18.223
    - type: precision_at_5
      value: 13.050999999999998
    - type: recall_at_1
      value: 28.126
    - type: recall_at_10
      value: 54.952
    - type: recall_at_100
      value: 78.375
    - type: recall_at_1000
      value: 93.29899999999999
    - type: recall_at_3
      value: 41.714
    - type: recall_at_5
      value: 47.635
    - type: map_at_1
      value: 25.957
    - type: map_at_10
      value: 34.749
    - type: map_at_100
      value: 35.929
    - type: map_at_1000
      value: 36.043
    - type: map_at_3
      value: 31.947
    - type: map_at_5
      value: 33.575
    - type: mrr_at_1
      value: 32.078
    - type: mrr_at_10
      value: 39.844
    - type: mrr_at_100
      value: 40.71
    - type: mrr_at_1000
      value: 40.77
    - type: mrr_at_3
      value: 37.386
    - type: mrr_at_5
      value: 38.83
    - type: ndcg_at_1
      value: 32.078
    - type: ndcg_at_10
      value: 39.97
    - type: ndcg_at_100
      value: 45.254
    - type: ndcg_at_1000
      value: 47.818
    - type: ndcg_at_3
      value: 35.453
    - type: ndcg_at_5
      value: 37.631
    - type: precision_at_1
      value: 32.078
    - type: precision_at_10
      value: 7.158
    - type: precision_at_100
      value: 1.126
    - type: precision_at_1000
      value: 0.153
    - type: precision_at_3
      value: 16.743
    - type: precision_at_5
      value: 11.872
    - type: recall_at_1
      value: 25.957
    - type: recall_at_10
      value: 50.583
    - type: recall_at_100
      value: 73.593
    - type: recall_at_1000
      value: 91.23599999999999
    - type: recall_at_3
      value: 37.651
    - type: recall_at_5
      value: 43.626
    - type: map_at_1
      value: 27.1505
    - type: map_at_10
      value: 34.844833333333334
    - type: map_at_100
      value: 35.95216666666667
    - type: map_at_1000
      value: 36.06675
    - type: map_at_3
      value: 32.41975
    - type: map_at_5
      value: 33.74233333333333
    - type: mrr_at_1
      value: 31.923666666666662
    - type: mrr_at_10
      value: 38.87983333333334
    - type: mrr_at_100
      value: 39.706250000000004
    - type: mrr_at_1000
      value: 39.76708333333333
    - type: mrr_at_3
      value: 36.72008333333333
    - type: mrr_at_5
      value: 37.96933333333334
    - type: ndcg_at_1
      value: 31.923666666666662
    - type: ndcg_at_10
      value: 39.44258333333334
    - type: ndcg_at_100
      value: 44.31475
    - type: ndcg_at_1000
      value: 46.75
    - type: ndcg_at_3
      value: 35.36299999999999
    - type: ndcg_at_5
      value: 37.242333333333335
    - type: precision_at_1
      value: 31.923666666666662
    - type: precision_at_10
      value: 6.643333333333333
    - type: precision_at_100
      value: 1.0612499999999998
    - type: precision_at_1000
      value: 0.14575
    - type: precision_at_3
      value: 15.875250000000001
    - type: precision_at_5
      value: 11.088916666666664
    - type: recall_at_1
      value: 27.1505
    - type: recall_at_10
      value: 49.06349999999999
    - type: recall_at_100
      value: 70.60841666666666
    - type: recall_at_1000
      value: 87.72049999999999
    - type: recall_at_3
      value: 37.60575000000001
    - type: recall_at_5
      value: 42.511166666666675
    - type: map_at_1
      value: 25.101000000000003
    - type: map_at_10
      value: 30.147000000000002
    - type: map_at_100
      value: 30.98
    - type: map_at_1000
      value: 31.080000000000002
    - type: map_at_3
      value: 28.571
    - type: map_at_5
      value: 29.319
    - type: mrr_at_1
      value: 27.761000000000003
    - type: mrr_at_10
      value: 32.716
    - type: mrr_at_100
      value: 33.504
    - type: mrr_at_1000
      value: 33.574
    - type: mrr_at_3
      value: 31.135
    - type: mrr_at_5
      value: 32.032
    - type: ndcg_at_1
      value: 27.761000000000003
    - type: ndcg_at_10
      value: 33.358
    - type: ndcg_at_100
      value: 37.569
    - type: ndcg_at_1000
      value: 40.189
    - type: ndcg_at_3
      value: 30.291
    - type: ndcg_at_5
      value: 31.558000000000003
    - type: precision_at_1
      value: 27.761000000000003
    - type: precision_at_10
      value: 4.939
    - type: precision_at_100
      value: 0.759
    - type: precision_at_1000
      value: 0.106
    - type: precision_at_3
      value: 12.577
    - type: precision_at_5
      value: 8.497
    - type: recall_at_1
      value: 25.101000000000003
    - type: recall_at_10
      value: 40.739
    - type: recall_at_100
      value: 60.089999999999996
    - type: recall_at_1000
      value: 79.768
    - type: recall_at_3
      value: 32.16
    - type: recall_at_5
      value: 35.131
    - type: map_at_1
      value: 20.112
    - type: map_at_10
      value: 26.119999999999997
    - type: map_at_100
      value: 27.031
    - type: map_at_1000
      value: 27.150000000000002
    - type: map_at_3
      value: 24.230999999999998
    - type: map_at_5
      value: 25.15
    - type: mrr_at_1
      value: 24.535
    - type: mrr_at_10
      value: 30.198000000000004
    - type: mrr_at_100
      value: 30.975
    - type: mrr_at_1000
      value: 31.051000000000002
    - type: mrr_at_3
      value: 28.338
    - type: mrr_at_5
      value: 29.269000000000002
    - type: ndcg_at_1
      value: 24.535
    - type: ndcg_at_10
      value: 30.147000000000002
    - type: ndcg_at_100
      value: 34.544000000000004
    - type: ndcg_at_1000
      value: 37.512
    - type: ndcg_at_3
      value: 26.726
    - type: ndcg_at_5
      value: 28.046
    - type: precision_at_1
      value: 24.535
    - type: precision_at_10
      value: 5.179
    - type: precision_at_100
      value: 0.859
    - type: precision_at_1000
      value: 0.128
    - type: precision_at_3
      value: 12.159
    - type: precision_at_5
      value: 8.424
    - type: recall_at_1
      value: 20.112
    - type: recall_at_10
      value: 38.312000000000005
    - type: recall_at_100
      value: 58.406000000000006
    - type: recall_at_1000
      value: 79.863
    - type: recall_at_3
      value: 28.358
    - type: recall_at_5
      value: 31.973000000000003
    - type: map_at_1
      value: 27.111
    - type: map_at_10
      value: 34.096
    - type: map_at_100
      value: 35.181000000000004
    - type: map_at_1000
      value: 35.276
    - type: map_at_3
      value: 31.745
    - type: map_at_5
      value: 33.045
    - type: mrr_at_1
      value: 31.343
    - type: mrr_at_10
      value: 37.994
    - type: mrr_at_100
      value: 38.873000000000005
    - type: mrr_at_1000
      value: 38.934999999999995
    - type: mrr_at_3
      value: 35.743
    - type: mrr_at_5
      value: 37.077
    - type: ndcg_at_1
      value: 31.343
    - type: ndcg_at_10
      value: 38.572
    - type: ndcg_at_100
      value: 43.854
    - type: ndcg_at_1000
      value: 46.190999999999995
    - type: ndcg_at_3
      value: 34.247
    - type: ndcg_at_5
      value: 36.28
    - type: precision_at_1
      value: 31.343
    - type: precision_at_10
      value: 6.166
    - type: precision_at_100
      value: 1
    - type: precision_at_1000
      value: 0.13
    - type: precision_at_3
      value: 15.081
    - type: precision_at_5
      value: 10.428999999999998
    - type: recall_at_1
      value: 27.111
    - type: recall_at_10
      value: 48.422
    - type: recall_at_100
      value: 71.846
    - type: recall_at_1000
      value: 88.57000000000001
    - type: recall_at_3
      value: 36.435
    - type: recall_at_5
      value: 41.765
    - type: map_at_1
      value: 26.264
    - type: map_at_10
      value: 33.522
    - type: map_at_100
      value: 34.963
    - type: map_at_1000
      value: 35.175
    - type: map_at_3
      value: 31.366
    - type: map_at_5
      value: 32.621
    - type: mrr_at_1
      value: 31.028
    - type: mrr_at_10
      value: 37.230000000000004
    - type: mrr_at_100
      value: 38.149
    - type: mrr_at_1000
      value: 38.218
    - type: mrr_at_3
      value: 35.046
    - type: mrr_at_5
      value: 36.617
    - type: ndcg_at_1
      value: 31.028
    - type: ndcg_at_10
      value: 37.964999999999996
    - type: ndcg_at_100
      value: 43.342000000000006
    - type: ndcg_at_1000
      value: 46.471000000000004
    - type: ndcg_at_3
      value: 34.67
    - type: ndcg_at_5
      value: 36.458
    - type: precision_at_1
      value: 31.028
    - type: precision_at_10
      value: 6.937
    - type: precision_at_100
      value: 1.346
    - type: precision_at_1000
      value: 0.22799999999999998
    - type: precision_at_3
      value: 15.942
    - type: precision_at_5
      value: 11.462
    - type: recall_at_1
      value: 26.264
    - type: recall_at_10
      value: 45.571
    - type: recall_at_100
      value: 70.246
    - type: recall_at_1000
      value: 90.971
    - type: recall_at_3
      value: 36.276
    - type: recall_at_5
      value: 41.162
    - type: map_at_1
      value: 23.372999999999998
    - type: map_at_10
      value: 28.992
    - type: map_at_100
      value: 29.837999999999997
    - type: map_at_1000
      value: 29.939
    - type: map_at_3
      value: 26.999000000000002
    - type: map_at_5
      value: 28.044999999999998
    - type: mrr_at_1
      value: 25.692999999999998
    - type: mrr_at_10
      value: 30.984
    - type: mrr_at_100
      value: 31.799
    - type: mrr_at_1000
      value: 31.875999999999998
    - type: mrr_at_3
      value: 29.267
    - type: mrr_at_5
      value: 30.163
    - type: ndcg_at_1
      value: 25.692999999999998
    - type: ndcg_at_10
      value: 32.45
    - type: ndcg_at_100
      value: 37.103
    - type: ndcg_at_1000
      value: 39.678000000000004
    - type: ndcg_at_3
      value: 28.725
    - type: ndcg_at_5
      value: 30.351
    - type: precision_at_1
      value: 25.692999999999998
    - type: precision_at_10
      value: 4.806
    - type: precision_at_100
      value: 0.765
    - type: precision_at_1000
      value: 0.108
    - type: precision_at_3
      value: 11.768
    - type: precision_at_5
      value: 8.096
    - type: recall_at_1
      value: 23.372999999999998
    - type: recall_at_10
      value: 41.281
    - type: recall_at_100
      value: 63.465
    - type: recall_at_1000
      value: 82.575
    - type: recall_at_3
      value: 31.063000000000002
    - type: recall_at_5
      value: 34.991
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
      value: 8.821
    - type: map_at_10
      value: 15.383
    - type: map_at_100
      value: 17.244999999999997
    - type: map_at_1000
      value: 17.445
    - type: map_at_3
      value: 12.64
    - type: map_at_5
      value: 13.941999999999998
    - type: mrr_at_1
      value: 19.544
    - type: mrr_at_10
      value: 29.738999999999997
    - type: mrr_at_100
      value: 30.923000000000002
    - type: mrr_at_1000
      value: 30.969
    - type: mrr_at_3
      value: 26.384
    - type: mrr_at_5
      value: 28.199
    - type: ndcg_at_1
      value: 19.544
    - type: ndcg_at_10
      value: 22.398
    - type: ndcg_at_100
      value: 30.253999999999998
    - type: ndcg_at_1000
      value: 33.876
    - type: ndcg_at_3
      value: 17.473
    - type: ndcg_at_5
      value: 19.154
    - type: precision_at_1
      value: 19.544
    - type: precision_at_10
      value: 7.217999999999999
    - type: precision_at_100
      value: 1.564
    - type: precision_at_1000
      value: 0.22300000000000003
    - type: precision_at_3
      value: 13.225000000000001
    - type: precision_at_5
      value: 10.319
    - type: recall_at_1
      value: 8.821
    - type: recall_at_10
      value: 28.110000000000003
    - type: recall_at_100
      value: 55.64
    - type: recall_at_1000
      value: 75.964
    - type: recall_at_3
      value: 16.195
    - type: recall_at_5
      value: 20.678
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
      value: 9.344
    - type: map_at_10
      value: 20.301
    - type: map_at_100
      value: 28.709
    - type: map_at_1000
      value: 30.470999999999997
    - type: map_at_3
      value: 14.584
    - type: map_at_5
      value: 16.930999999999997
    - type: mrr_at_1
      value: 67.25
    - type: mrr_at_10
      value: 75.393
    - type: mrr_at_100
      value: 75.742
    - type: mrr_at_1000
      value: 75.75
    - type: mrr_at_3
      value: 73.958
    - type: mrr_at_5
      value: 74.883
    - type: ndcg_at_1
      value: 56.00000000000001
    - type: ndcg_at_10
      value: 42.394
    - type: ndcg_at_100
      value: 47.091
    - type: ndcg_at_1000
      value: 54.215
    - type: ndcg_at_3
      value: 46.995
    - type: ndcg_at_5
      value: 44.214999999999996
    - type: precision_at_1
      value: 67.25
    - type: precision_at_10
      value: 33.525
    - type: precision_at_100
      value: 10.67
    - type: precision_at_1000
      value: 2.221
    - type: precision_at_3
      value: 49.417
    - type: precision_at_5
      value: 42.15
    - type: recall_at_1
      value: 9.344
    - type: recall_at_10
      value: 25.209
    - type: recall_at_100
      value: 52.329
    - type: recall_at_1000
      value: 74.2
    - type: recall_at_3
      value: 15.699
    - type: recall_at_5
      value: 19.24
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
      value: 48.05
    - type: f1
      value: 43.06718139212933
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
      value: 46.452
    - type: map_at_10
      value: 58.825
    - type: map_at_100
      value: 59.372
    - type: map_at_1000
      value: 59.399
    - type: map_at_3
      value: 56.264
    - type: map_at_5
      value: 57.879999999999995
    - type: mrr_at_1
      value: 49.82
    - type: mrr_at_10
      value: 62.178999999999995
    - type: mrr_at_100
      value: 62.641999999999996
    - type: mrr_at_1000
      value: 62.658
    - type: mrr_at_3
      value: 59.706
    - type: mrr_at_5
      value: 61.283
    - type: ndcg_at_1
      value: 49.82
    - type: ndcg_at_10
      value: 65.031
    - type: ndcg_at_100
      value: 67.413
    - type: ndcg_at_1000
      value: 68.014
    - type: ndcg_at_3
      value: 60.084
    - type: ndcg_at_5
      value: 62.858000000000004
    - type: precision_at_1
      value: 49.82
    - type: precision_at_10
      value: 8.876000000000001
    - type: precision_at_100
      value: 1.018
    - type: precision_at_1000
      value: 0.109
    - type: precision_at_3
      value: 24.477
    - type: precision_at_5
      value: 16.208
    - type: recall_at_1
      value: 46.452
    - type: recall_at_10
      value: 80.808
    - type: recall_at_100
      value: 91.215
    - type: recall_at_1000
      value: 95.52000000000001
    - type: recall_at_3
      value: 67.62899999999999
    - type: recall_at_5
      value: 74.32900000000001
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
      value: 18.351
    - type: map_at_10
      value: 30.796
    - type: map_at_100
      value: 32.621
    - type: map_at_1000
      value: 32.799
    - type: map_at_3
      value: 26.491
    - type: map_at_5
      value: 28.933999999999997
    - type: mrr_at_1
      value: 36.265
    - type: mrr_at_10
      value: 45.556999999999995
    - type: mrr_at_100
      value: 46.323
    - type: mrr_at_1000
      value: 46.359
    - type: mrr_at_3
      value: 42.695
    - type: mrr_at_5
      value: 44.324000000000005
    - type: ndcg_at_1
      value: 36.265
    - type: ndcg_at_10
      value: 38.558
    - type: ndcg_at_100
      value: 45.18
    - type: ndcg_at_1000
      value: 48.292
    - type: ndcg_at_3
      value: 34.204
    - type: ndcg_at_5
      value: 35.735
    - type: precision_at_1
      value: 36.265
    - type: precision_at_10
      value: 10.879999999999999
    - type: precision_at_100
      value: 1.77
    - type: precision_at_1000
      value: 0.234
    - type: precision_at_3
      value: 23.044999999999998
    - type: precision_at_5
      value: 17.253
    - type: recall_at_1
      value: 18.351
    - type: recall_at_10
      value: 46.116
    - type: recall_at_100
      value: 70.786
    - type: recall_at_1000
      value: 89.46300000000001
    - type: recall_at_3
      value: 31.404
    - type: recall_at_5
      value: 37.678
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
      value: 36.847
    - type: map_at_10
      value: 54.269999999999996
    - type: map_at_100
      value: 55.152
    - type: map_at_1000
      value: 55.223
    - type: map_at_3
      value: 51.166
    - type: map_at_5
      value: 53.055
    - type: mrr_at_1
      value: 73.693
    - type: mrr_at_10
      value: 79.975
    - type: mrr_at_100
      value: 80.202
    - type: mrr_at_1000
      value: 80.214
    - type: mrr_at_3
      value: 78.938
    - type: mrr_at_5
      value: 79.595
    - type: ndcg_at_1
      value: 73.693
    - type: ndcg_at_10
      value: 63.334999999999994
    - type: ndcg_at_100
      value: 66.452
    - type: ndcg_at_1000
      value: 67.869
    - type: ndcg_at_3
      value: 58.829
    - type: ndcg_at_5
      value: 61.266
    - type: precision_at_1
      value: 73.693
    - type: precision_at_10
      value: 13.122
    - type: precision_at_100
      value: 1.5559999999999998
    - type: precision_at_1000
      value: 0.174
    - type: precision_at_3
      value: 37.083
    - type: precision_at_5
      value: 24.169999999999998
    - type: recall_at_1
      value: 36.847
    - type: recall_at_10
      value: 65.61099999999999
    - type: recall_at_100
      value: 77.792
    - type: recall_at_1000
      value: 87.17099999999999
    - type: recall_at_3
      value: 55.625
    - type: recall_at_5
      value: 60.425
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
      value: 82.1096
    - type: ap
      value: 76.67089212843918
    - type: f1
      value: 82.03535056754939
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
      value: 24.465
    - type: map_at_10
      value: 37.072
    - type: map_at_100
      value: 38.188
    - type: map_at_1000
      value: 38.232
    - type: map_at_3
      value: 33.134
    - type: map_at_5
      value: 35.453
    - type: mrr_at_1
      value: 25.142999999999997
    - type: mrr_at_10
      value: 37.669999999999995
    - type: mrr_at_100
      value: 38.725
    - type: mrr_at_1000
      value: 38.765
    - type: mrr_at_3
      value: 33.82
    - type: mrr_at_5
      value: 36.111
    - type: ndcg_at_1
      value: 25.142999999999997
    - type: ndcg_at_10
      value: 44.054
    - type: ndcg_at_100
      value: 49.364000000000004
    - type: ndcg_at_1000
      value: 50.456
    - type: ndcg_at_3
      value: 36.095
    - type: ndcg_at_5
      value: 40.23
    - type: precision_at_1
      value: 25.142999999999997
    - type: precision_at_10
      value: 6.845
    - type: precision_at_100
      value: 0.95
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 15.204999999999998
    - type: precision_at_5
      value: 11.221
    - type: recall_at_1
      value: 24.465
    - type: recall_at_10
      value: 65.495
    - type: recall_at_100
      value: 89.888
    - type: recall_at_1000
      value: 98.165
    - type: recall_at_3
      value: 43.964
    - type: recall_at_5
      value: 53.891
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
      value: 93.86228910168718
    - type: f1
      value: 93.69177113259104
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
      value: 76.3999088007296
    - type: f1
      value: 58.96668664333438
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
      value: 73.21788836583727
    - type: f1
      value: 71.4545936552952
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
      value: 77.39071956960323
    - type: f1
      value: 77.12398952847603
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
      value: 32.255379528166955
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
      value: 29.66423362872814
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
      value: 30.782211620375964
    - type: mrr
      value: 31.773479703044956
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
      value: 5.863
    - type: map_at_10
      value: 13.831
    - type: map_at_100
      value: 17.534
    - type: map_at_1000
      value: 19.012
    - type: map_at_3
      value: 10.143
    - type: map_at_5
      value: 12.034
    - type: mrr_at_1
      value: 46.749
    - type: mrr_at_10
      value: 55.376999999999995
    - type: mrr_at_100
      value: 56.009
    - type: mrr_at_1000
      value: 56.042
    - type: mrr_at_3
      value: 53.30200000000001
    - type: mrr_at_5
      value: 54.85
    - type: ndcg_at_1
      value: 44.582
    - type: ndcg_at_10
      value: 36.07
    - type: ndcg_at_100
      value: 33.39
    - type: ndcg_at_1000
      value: 41.884
    - type: ndcg_at_3
      value: 41.441
    - type: ndcg_at_5
      value: 39.861000000000004
    - type: precision_at_1
      value: 46.129999999999995
    - type: precision_at_10
      value: 26.594
    - type: precision_at_100
      value: 8.365
    - type: precision_at_1000
      value: 2.1260000000000003
    - type: precision_at_3
      value: 39.009
    - type: precision_at_5
      value: 34.861
    - type: recall_at_1
      value: 5.863
    - type: recall_at_10
      value: 17.961
    - type: recall_at_100
      value: 34.026
    - type: recall_at_1000
      value: 64.46499999999999
    - type: recall_at_3
      value: 11.242
    - type: recall_at_5
      value: 14.493
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
      value: 38.601
    - type: map_at_10
      value: 55.293000000000006
    - type: map_at_100
      value: 56.092
    - type: map_at_1000
      value: 56.111999999999995
    - type: map_at_3
      value: 51.269
    - type: map_at_5
      value: 53.787
    - type: mrr_at_1
      value: 43.221
    - type: mrr_at_10
      value: 57.882999999999996
    - type: mrr_at_100
      value: 58.408
    - type: mrr_at_1000
      value: 58.421
    - type: mrr_at_3
      value: 54.765
    - type: mrr_at_5
      value: 56.809
    - type: ndcg_at_1
      value: 43.221
    - type: ndcg_at_10
      value: 62.858999999999995
    - type: ndcg_at_100
      value: 65.987
    - type: ndcg_at_1000
      value: 66.404
    - type: ndcg_at_3
      value: 55.605000000000004
    - type: ndcg_at_5
      value: 59.723000000000006
    - type: precision_at_1
      value: 43.221
    - type: precision_at_10
      value: 9.907
    - type: precision_at_100
      value: 1.169
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 25.019000000000002
    - type: precision_at_5
      value: 17.474
    - type: recall_at_1
      value: 38.601
    - type: recall_at_10
      value: 82.966
    - type: recall_at_100
      value: 96.154
    - type: recall_at_1000
      value: 99.223
    - type: recall_at_3
      value: 64.603
    - type: recall_at_5
      value: 73.97200000000001
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
      value: 70.77
    - type: map_at_10
      value: 84.429
    - type: map_at_100
      value: 85.04599999999999
    - type: map_at_1000
      value: 85.065
    - type: map_at_3
      value: 81.461
    - type: map_at_5
      value: 83.316
    - type: mrr_at_1
      value: 81.51
    - type: mrr_at_10
      value: 87.52799999999999
    - type: mrr_at_100
      value: 87.631
    - type: mrr_at_1000
      value: 87.632
    - type: mrr_at_3
      value: 86.533
    - type: mrr_at_5
      value: 87.214
    - type: ndcg_at_1
      value: 81.47999999999999
    - type: ndcg_at_10
      value: 88.181
    - type: ndcg_at_100
      value: 89.39200000000001
    - type: ndcg_at_1000
      value: 89.52
    - type: ndcg_at_3
      value: 85.29299999999999
    - type: ndcg_at_5
      value: 86.88
    - type: precision_at_1
      value: 81.47999999999999
    - type: precision_at_10
      value: 13.367
    - type: precision_at_100
      value: 1.5230000000000001
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.227
    - type: precision_at_5
      value: 24.494
    - type: recall_at_1
      value: 70.77
    - type: recall_at_10
      value: 95.199
    - type: recall_at_100
      value: 99.37700000000001
    - type: recall_at_1000
      value: 99.973
    - type: recall_at_3
      value: 86.895
    - type: recall_at_5
      value: 91.396
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
      value: 50.686353396858344
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
      value: 61.3664675312921
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
      value: 4.7379999999999995
    - type: map_at_10
      value: 12.01
    - type: map_at_100
      value: 14.02
    - type: map_at_1000
      value: 14.310999999999998
    - type: map_at_3
      value: 8.459
    - type: map_at_5
      value: 10.281
    - type: mrr_at_1
      value: 23.3
    - type: mrr_at_10
      value: 34.108
    - type: mrr_at_100
      value: 35.217
    - type: mrr_at_1000
      value: 35.272
    - type: mrr_at_3
      value: 30.833
    - type: mrr_at_5
      value: 32.768
    - type: ndcg_at_1
      value: 23.3
    - type: ndcg_at_10
      value: 20.116999999999997
    - type: ndcg_at_100
      value: 27.961000000000002
    - type: ndcg_at_1000
      value: 33.149
    - type: ndcg_at_3
      value: 18.902
    - type: ndcg_at_5
      value: 16.742
    - type: precision_at_1
      value: 23.3
    - type: precision_at_10
      value: 10.47
    - type: precision_at_100
      value: 2.177
    - type: precision_at_1000
      value: 0.34299999999999997
    - type: precision_at_3
      value: 17.567
    - type: precision_at_5
      value: 14.78
    - type: recall_at_1
      value: 4.7379999999999995
    - type: recall_at_10
      value: 21.221999999999998
    - type: recall_at_100
      value: 44.242
    - type: recall_at_1000
      value: 69.652
    - type: recall_at_3
      value: 10.688
    - type: recall_at_5
      value: 14.982999999999999
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
      value: 84.84572946827069
    - type: cos_sim_spearman
      value: 80.48508130408966
    - type: euclidean_pearson
      value: 82.0481530027767
    - type: euclidean_spearman
      value: 80.45902876782752
    - type: manhattan_pearson
      value: 82.03728222483326
    - type: manhattan_spearman
      value: 80.45684282911755
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
      value: 84.33476464677516
    - type: cos_sim_spearman
      value: 75.93057758003266
    - type: euclidean_pearson
      value: 80.89685744015691
    - type: euclidean_spearman
      value: 76.29929953441706
    - type: manhattan_pearson
      value: 80.91391345459995
    - type: manhattan_spearman
      value: 76.31985463110914
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
      value: 84.63686106359005
    - type: cos_sim_spearman
      value: 85.22240034668202
    - type: euclidean_pearson
      value: 84.6074814189106
    - type: euclidean_spearman
      value: 85.17169644755828
    - type: manhattan_pearson
      value: 84.48329306239368
    - type: manhattan_spearman
      value: 85.0086508544768
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
      value: 82.95455774064745
    - type: cos_sim_spearman
      value: 80.54074646118492
    - type: euclidean_pearson
      value: 81.79598955554704
    - type: euclidean_spearman
      value: 80.55837617606814
    - type: manhattan_pearson
      value: 81.78213797905386
    - type: manhattan_spearman
      value: 80.5666746878273
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
      value: 87.92813309124739
    - type: cos_sim_spearman
      value: 88.81459873052108
    - type: euclidean_pearson
      value: 88.21193118930564
    - type: euclidean_spearman
      value: 88.87072745043731
    - type: manhattan_pearson
      value: 88.22576929706727
    - type: manhattan_spearman
      value: 88.8867671095791
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
      value: 83.6881529671839
    - type: cos_sim_spearman
      value: 85.2807092969554
    - type: euclidean_pearson
      value: 84.62334178652704
    - type: euclidean_spearman
      value: 85.2116373296784
    - type: manhattan_pearson
      value: 84.54948211541777
    - type: manhattan_spearman
      value: 85.10737722637882
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
      value: 88.55963694458408
    - type: cos_sim_spearman
      value: 89.36731628848683
    - type: euclidean_pearson
      value: 89.64975952985465
    - type: euclidean_spearman
      value: 89.29689484033007
    - type: manhattan_pearson
      value: 89.61234491713135
    - type: manhattan_spearman
      value: 89.20302520255782
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
      value: 62.411800961903886
    - type: cos_sim_spearman
      value: 62.99105515749963
    - type: euclidean_pearson
      value: 65.29826669549443
    - type: euclidean_spearman
      value: 63.29880964105775
    - type: manhattan_pearson
      value: 65.00126190601183
    - type: manhattan_spearman
      value: 63.32011025899179
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
      value: 85.83498531837608
    - type: cos_sim_spearman
      value: 87.21366640615442
    - type: euclidean_pearson
      value: 86.74764288798261
    - type: euclidean_spearman
      value: 87.06060470780834
    - type: manhattan_pearson
      value: 86.65971223951476
    - type: manhattan_spearman
      value: 86.99814399831457
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
      value: 83.94448463485881
    - type: mrr
      value: 95.36291867174221
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
      value: 59.928000000000004
    - type: map_at_10
      value: 68.577
    - type: map_at_100
      value: 69.35900000000001
    - type: map_at_1000
      value: 69.37299999999999
    - type: map_at_3
      value: 66.217
    - type: map_at_5
      value: 67.581
    - type: mrr_at_1
      value: 63
    - type: mrr_at_10
      value: 69.994
    - type: mrr_at_100
      value: 70.553
    - type: mrr_at_1000
      value: 70.56700000000001
    - type: mrr_at_3
      value: 68.167
    - type: mrr_at_5
      value: 69.11699999999999
    - type: ndcg_at_1
      value: 63
    - type: ndcg_at_10
      value: 72.58
    - type: ndcg_at_100
      value: 75.529
    - type: ndcg_at_1000
      value: 76.009
    - type: ndcg_at_3
      value: 68.523
    - type: ndcg_at_5
      value: 70.301
    - type: precision_at_1
      value: 63
    - type: precision_at_10
      value: 9.333
    - type: precision_at_100
      value: 1.09
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 26.444000000000003
    - type: precision_at_5
      value: 17.067
    - type: recall_at_1
      value: 59.928000000000004
    - type: recall_at_10
      value: 83.544
    - type: recall_at_100
      value: 96
    - type: recall_at_1000
      value: 100
    - type: recall_at_3
      value: 72.072
    - type: recall_at_5
      value: 76.683
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
      value: 99.82178217821782
    - type: cos_sim_ap
      value: 95.41507679819003
    - type: cos_sim_f1
      value: 90.9456740442656
    - type: cos_sim_precision
      value: 91.49797570850203
    - type: cos_sim_recall
      value: 90.4
    - type: dot_accuracy
      value: 99.77227722772277
    - type: dot_ap
      value: 92.50123869445967
    - type: dot_f1
      value: 88.18414322250638
    - type: dot_precision
      value: 90.26178010471205
    - type: dot_recall
      value: 86.2
    - type: euclidean_accuracy
      value: 99.81782178217821
    - type: euclidean_ap
      value: 95.3935066749006
    - type: euclidean_f1
      value: 90.66128218071681
    - type: euclidean_precision
      value: 91.53924566768603
    - type: euclidean_recall
      value: 89.8
    - type: manhattan_accuracy
      value: 99.81881188118813
    - type: manhattan_ap
      value: 95.39767454613512
    - type: manhattan_f1
      value: 90.62019477191186
    - type: manhattan_precision
      value: 92.95478443743428
    - type: manhattan_recall
      value: 88.4
    - type: max_accuracy
      value: 99.82178217821782
    - type: max_ap
      value: 95.41507679819003
    - type: max_f1
      value: 90.9456740442656
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
      value: 64.96313921233748
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
      value: 33.602625720956745
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
      value: 51.32659230651731
    - type: mrr
      value: 52.33861726508785
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
      value: 31.01587644214203
    - type: cos_sim_spearman
      value: 30.974306908731013
    - type: dot_pearson
      value: 29.83339853838187
    - type: dot_spearman
      value: 30.07761671934048
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
      value: 1.9539999999999997
    - type: map_at_100
      value: 11.437
    - type: map_at_1000
      value: 27.861000000000004
    - type: map_at_3
      value: 0.6479999999999999
    - type: map_at_5
      value: 1.0410000000000001
    - type: mrr_at_1
      value: 84
    - type: mrr_at_10
      value: 90.333
    - type: mrr_at_100
      value: 90.333
    - type: mrr_at_1000
      value: 90.333
    - type: mrr_at_3
      value: 90.333
    - type: mrr_at_5
      value: 90.333
    - type: ndcg_at_1
      value: 80
    - type: ndcg_at_10
      value: 78.31700000000001
    - type: ndcg_at_100
      value: 59.396
    - type: ndcg_at_1000
      value: 52.733
    - type: ndcg_at_3
      value: 81.46900000000001
    - type: ndcg_at_5
      value: 80.74
    - type: precision_at_1
      value: 84
    - type: precision_at_10
      value: 84
    - type: precision_at_100
      value: 60.980000000000004
    - type: precision_at_1000
      value: 23.432
    - type: precision_at_3
      value: 87.333
    - type: precision_at_5
      value: 86.8
    - type: recall_at_1
      value: 0.22
    - type: recall_at_10
      value: 2.156
    - type: recall_at_100
      value: 14.557999999999998
    - type: recall_at_1000
      value: 49.553999999999995
    - type: recall_at_3
      value: 0.685
    - type: recall_at_5
      value: 1.121
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
      value: 3.373
    - type: map_at_10
      value: 11.701
    - type: map_at_100
      value: 17.144000000000002
    - type: map_at_1000
      value: 18.624
    - type: map_at_3
      value: 6.552
    - type: map_at_5
      value: 9.372
    - type: mrr_at_1
      value: 38.775999999999996
    - type: mrr_at_10
      value: 51.975
    - type: mrr_at_100
      value: 52.873999999999995
    - type: mrr_at_1000
      value: 52.873999999999995
    - type: mrr_at_3
      value: 47.619
    - type: mrr_at_5
      value: 50.578
    - type: ndcg_at_1
      value: 36.735
    - type: ndcg_at_10
      value: 27.212999999999997
    - type: ndcg_at_100
      value: 37.245
    - type: ndcg_at_1000
      value: 48.602000000000004
    - type: ndcg_at_3
      value: 30.916
    - type: ndcg_at_5
      value: 30.799
    - type: precision_at_1
      value: 38.775999999999996
    - type: precision_at_10
      value: 23.469
    - type: precision_at_100
      value: 7.327
    - type: precision_at_1000
      value: 1.486
    - type: precision_at_3
      value: 31.973000000000003
    - type: precision_at_5
      value: 32.245000000000005
    - type: recall_at_1
      value: 3.373
    - type: recall_at_10
      value: 17.404
    - type: recall_at_100
      value: 46.105000000000004
    - type: recall_at_1000
      value: 80.35
    - type: recall_at_3
      value: 7.4399999999999995
    - type: recall_at_5
      value: 12.183
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
      value: 70.5592
    - type: ap
      value: 14.330910591410134
    - type: f1
      value: 54.45745186286521
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
      value: 61.20543293718167
    - type: f1
      value: 61.45365480309872
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
      value: 43.81162998944145
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
      value: 86.69011146212075
    - type: cos_sim_ap
      value: 76.09792353652536
    - type: cos_sim_f1
      value: 70.10202763786646
    - type: cos_sim_precision
      value: 68.65671641791045
    - type: cos_sim_recall
      value: 71.60949868073878
    - type: dot_accuracy
      value: 85.33110806461227
    - type: dot_ap
      value: 70.19304383327554
    - type: dot_f1
      value: 67.22494202525122
    - type: dot_precision
      value: 65.6847935548842
    - type: dot_recall
      value: 68.83905013192611
    - type: euclidean_accuracy
      value: 86.5410979316922
    - type: euclidean_ap
      value: 75.91906915651882
    - type: euclidean_f1
      value: 69.6798975672215
    - type: euclidean_precision
      value: 67.6865671641791
    - type: euclidean_recall
      value: 71.79419525065963
    - type: manhattan_accuracy
      value: 86.60070334386363
    - type: manhattan_ap
      value: 75.94617413885031
    - type: manhattan_f1
      value: 69.52689565780946
    - type: manhattan_precision
      value: 68.3312101910828
    - type: manhattan_recall
      value: 70.76517150395777
    - type: max_accuracy
      value: 86.69011146212075
    - type: max_ap
      value: 76.09792353652536
    - type: max_f1
      value: 70.10202763786646
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
      value: 89.25951798812434
    - type: cos_sim_ap
      value: 86.31476416599727
    - type: cos_sim_f1
      value: 78.52709971038477
    - type: cos_sim_precision
      value: 76.7629972792117
    - type: cos_sim_recall
      value: 80.37419156144134
    - type: dot_accuracy
      value: 88.03896456708192
    - type: dot_ap
      value: 83.26963599196237
    - type: dot_f1
      value: 76.72696459492317
    - type: dot_precision
      value: 73.56411162133521
    - type: dot_recall
      value: 80.17400677548507
    - type: euclidean_accuracy
      value: 89.21682772538519
    - type: euclidean_ap
      value: 86.29306071289969
    - type: euclidean_f1
      value: 78.40827030519554
    - type: euclidean_precision
      value: 77.42250243939053
    - type: euclidean_recall
      value: 79.41946412072683
    - type: manhattan_accuracy
      value: 89.22458959133776
    - type: manhattan_ap
      value: 86.2901934710645
    - type: manhattan_f1
      value: 78.54211378440453
    - type: manhattan_precision
      value: 76.85505858079729
    - type: manhattan_recall
      value: 80.30489682784109
    - type: max_accuracy
      value: 89.25951798812434
    - type: max_ap
      value: 86.31476416599727
    - type: max_f1
      value: 78.54211378440453
---

## E5-large

**News (May 2023): please switch to [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2), which has better performance and same method of usage.**

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

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large')
model = AutoModel.from_pretrained('intfloat/e5-large')

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
model = SentenceTransformer('intfloat/e5-large')
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
