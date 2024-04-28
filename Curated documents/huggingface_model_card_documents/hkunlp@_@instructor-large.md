---
language: en
license: apache-2.0
tags:
- text-embedding
- embeddings
- information-retrieval
- beir
- text-classification
- language-model
- text-clustering
- text-semantic-similarity
- text-evaluation
- prompt-retrieval
- text-reranking
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- t5
- English
- Sentence Similarity
- natural_questions
- ms_marco
- fever
- hotpot_qa
- mteb
pipeline_tag: sentence-similarity
inference: false
model-index:
- name: INSTRUCTOR
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
      value: 88.13432835820896
    - type: ap
      value: 59.298209334395665
    - type: f1
      value: 83.31769058643586
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
      value: 91.526375
    - type: ap
      value: 88.16327709705504
    - type: f1
      value: 91.51095801287843
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
      value: 47.856
    - type: f1
      value: 45.41490917650942
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
      value: 31.223
    - type: map_at_10
      value: 47.947
    - type: map_at_100
      value: 48.742000000000004
    - type: map_at_1000
      value: 48.745
    - type: map_at_3
      value: 43.137
    - type: map_at_5
      value: 45.992
    - type: mrr_at_1
      value: 32.432
    - type: mrr_at_10
      value: 48.4
    - type: mrr_at_100
      value: 49.202
    - type: mrr_at_1000
      value: 49.205
    - type: mrr_at_3
      value: 43.551
    - type: mrr_at_5
      value: 46.467999999999996
    - type: ndcg_at_1
      value: 31.223
    - type: ndcg_at_10
      value: 57.045
    - type: ndcg_at_100
      value: 60.175
    - type: ndcg_at_1000
      value: 60.233000000000004
    - type: ndcg_at_3
      value: 47.171
    - type: ndcg_at_5
      value: 52.322
    - type: precision_at_1
      value: 31.223
    - type: precision_at_10
      value: 8.599
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 19.63
    - type: precision_at_5
      value: 14.282
    - type: recall_at_1
      value: 31.223
    - type: recall_at_10
      value: 85.989
    - type: recall_at_100
      value: 99.075
    - type: recall_at_1000
      value: 99.502
    - type: recall_at_3
      value: 58.89
    - type: recall_at_5
      value: 71.408
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
      value: 43.1621946393635
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
      value: 32.56417132407894
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
      value: 64.29539304390207
    - type: mrr
      value: 76.44484017060196
  - task:
      type: STS
    dataset:
      name: MTEB BIOSSES
      type: mteb/biosses-sts
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_spearman
      value: 84.38746499431112
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
      value: 78.51298701298701
    - type: f1
      value: 77.49041754069235
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
      value: 37.61848554098577
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
      value: 31.32623280148178
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
      value: 35.803000000000004
    - type: map_at_10
      value: 48.848
    - type: map_at_100
      value: 50.5
    - type: map_at_1000
      value: 50.602999999999994
    - type: map_at_3
      value: 45.111000000000004
    - type: map_at_5
      value: 47.202
    - type: mrr_at_1
      value: 44.635000000000005
    - type: mrr_at_10
      value: 55.593
    - type: mrr_at_100
      value: 56.169999999999995
    - type: mrr_at_1000
      value: 56.19499999999999
    - type: mrr_at_3
      value: 53.361999999999995
    - type: mrr_at_5
      value: 54.806999999999995
    - type: ndcg_at_1
      value: 44.635000000000005
    - type: ndcg_at_10
      value: 55.899
    - type: ndcg_at_100
      value: 60.958
    - type: ndcg_at_1000
      value: 62.302
    - type: ndcg_at_3
      value: 51.051
    - type: ndcg_at_5
      value: 53.351000000000006
    - type: precision_at_1
      value: 44.635000000000005
    - type: precision_at_10
      value: 10.786999999999999
    - type: precision_at_100
      value: 1.6580000000000001
    - type: precision_at_1000
      value: 0.213
    - type: precision_at_3
      value: 24.893
    - type: precision_at_5
      value: 17.740000000000002
    - type: recall_at_1
      value: 35.803000000000004
    - type: recall_at_10
      value: 68.657
    - type: recall_at_100
      value: 89.77199999999999
    - type: recall_at_1000
      value: 97.67
    - type: recall_at_3
      value: 54.066
    - type: recall_at_5
      value: 60.788
    - type: map_at_1
      value: 33.706
    - type: map_at_10
      value: 44.896
    - type: map_at_100
      value: 46.299
    - type: map_at_1000
      value: 46.44
    - type: map_at_3
      value: 41.721000000000004
    - type: map_at_5
      value: 43.486000000000004
    - type: mrr_at_1
      value: 41.592
    - type: mrr_at_10
      value: 50.529
    - type: mrr_at_100
      value: 51.22
    - type: mrr_at_1000
      value: 51.258
    - type: mrr_at_3
      value: 48.205999999999996
    - type: mrr_at_5
      value: 49.528
    - type: ndcg_at_1
      value: 41.592
    - type: ndcg_at_10
      value: 50.77199999999999
    - type: ndcg_at_100
      value: 55.383
    - type: ndcg_at_1000
      value: 57.288
    - type: ndcg_at_3
      value: 46.324
    - type: ndcg_at_5
      value: 48.346000000000004
    - type: precision_at_1
      value: 41.592
    - type: precision_at_10
      value: 9.516
    - type: precision_at_100
      value: 1.541
    - type: precision_at_1000
      value: 0.2
    - type: precision_at_3
      value: 22.399
    - type: precision_at_5
      value: 15.770999999999999
    - type: recall_at_1
      value: 33.706
    - type: recall_at_10
      value: 61.353
    - type: recall_at_100
      value: 80.182
    - type: recall_at_1000
      value: 91.896
    - type: recall_at_3
      value: 48.204
    - type: recall_at_5
      value: 53.89699999999999
    - type: map_at_1
      value: 44.424
    - type: map_at_10
      value: 57.169000000000004
    - type: map_at_100
      value: 58.202
    - type: map_at_1000
      value: 58.242000000000004
    - type: map_at_3
      value: 53.825
    - type: map_at_5
      value: 55.714
    - type: mrr_at_1
      value: 50.470000000000006
    - type: mrr_at_10
      value: 60.489000000000004
    - type: mrr_at_100
      value: 61.096
    - type: mrr_at_1000
      value: 61.112
    - type: mrr_at_3
      value: 58.192
    - type: mrr_at_5
      value: 59.611999999999995
    - type: ndcg_at_1
      value: 50.470000000000006
    - type: ndcg_at_10
      value: 63.071999999999996
    - type: ndcg_at_100
      value: 66.964
    - type: ndcg_at_1000
      value: 67.659
    - type: ndcg_at_3
      value: 57.74399999999999
    - type: ndcg_at_5
      value: 60.367000000000004
    - type: precision_at_1
      value: 50.470000000000006
    - type: precision_at_10
      value: 10.019
    - type: precision_at_100
      value: 1.29
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 25.558999999999997
    - type: precision_at_5
      value: 17.467
    - type: recall_at_1
      value: 44.424
    - type: recall_at_10
      value: 77.02
    - type: recall_at_100
      value: 93.738
    - type: recall_at_1000
      value: 98.451
    - type: recall_at_3
      value: 62.888
    - type: recall_at_5
      value: 69.138
    - type: map_at_1
      value: 26.294
    - type: map_at_10
      value: 34.503
    - type: map_at_100
      value: 35.641
    - type: map_at_1000
      value: 35.724000000000004
    - type: map_at_3
      value: 31.753999999999998
    - type: map_at_5
      value: 33.190999999999995
    - type: mrr_at_1
      value: 28.362
    - type: mrr_at_10
      value: 36.53
    - type: mrr_at_100
      value: 37.541000000000004
    - type: mrr_at_1000
      value: 37.602000000000004
    - type: mrr_at_3
      value: 33.917
    - type: mrr_at_5
      value: 35.358000000000004
    - type: ndcg_at_1
      value: 28.362
    - type: ndcg_at_10
      value: 39.513999999999996
    - type: ndcg_at_100
      value: 44.815
    - type: ndcg_at_1000
      value: 46.839
    - type: ndcg_at_3
      value: 34.02
    - type: ndcg_at_5
      value: 36.522
    - type: precision_at_1
      value: 28.362
    - type: precision_at_10
      value: 6.101999999999999
    - type: precision_at_100
      value: 0.9129999999999999
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 14.161999999999999
    - type: precision_at_5
      value: 9.966
    - type: recall_at_1
      value: 26.294
    - type: recall_at_10
      value: 53.098
    - type: recall_at_100
      value: 76.877
    - type: recall_at_1000
      value: 91.834
    - type: recall_at_3
      value: 38.266
    - type: recall_at_5
      value: 44.287
    - type: map_at_1
      value: 16.407
    - type: map_at_10
      value: 25.185999999999996
    - type: map_at_100
      value: 26.533
    - type: map_at_1000
      value: 26.657999999999998
    - type: map_at_3
      value: 22.201999999999998
    - type: map_at_5
      value: 23.923
    - type: mrr_at_1
      value: 20.522000000000002
    - type: mrr_at_10
      value: 29.522
    - type: mrr_at_100
      value: 30.644
    - type: mrr_at_1000
      value: 30.713
    - type: mrr_at_3
      value: 26.679000000000002
    - type: mrr_at_5
      value: 28.483000000000004
    - type: ndcg_at_1
      value: 20.522000000000002
    - type: ndcg_at_10
      value: 30.656
    - type: ndcg_at_100
      value: 36.864999999999995
    - type: ndcg_at_1000
      value: 39.675
    - type: ndcg_at_3
      value: 25.319000000000003
    - type: ndcg_at_5
      value: 27.992
    - type: precision_at_1
      value: 20.522000000000002
    - type: precision_at_10
      value: 5.795999999999999
    - type: precision_at_100
      value: 1.027
    - type: precision_at_1000
      value: 0.13999999999999999
    - type: precision_at_3
      value: 12.396
    - type: precision_at_5
      value: 9.328
    - type: recall_at_1
      value: 16.407
    - type: recall_at_10
      value: 43.164
    - type: recall_at_100
      value: 69.695
    - type: recall_at_1000
      value: 89.41900000000001
    - type: recall_at_3
      value: 28.634999999999998
    - type: recall_at_5
      value: 35.308
    - type: map_at_1
      value: 30.473
    - type: map_at_10
      value: 41.676
    - type: map_at_100
      value: 43.120999999999995
    - type: map_at_1000
      value: 43.230000000000004
    - type: map_at_3
      value: 38.306000000000004
    - type: map_at_5
      value: 40.355999999999995
    - type: mrr_at_1
      value: 37.536
    - type: mrr_at_10
      value: 47.643
    - type: mrr_at_100
      value: 48.508
    - type: mrr_at_1000
      value: 48.551
    - type: mrr_at_3
      value: 45.348
    - type: mrr_at_5
      value: 46.744
    - type: ndcg_at_1
      value: 37.536
    - type: ndcg_at_10
      value: 47.823
    - type: ndcg_at_100
      value: 53.395
    - type: ndcg_at_1000
      value: 55.271
    - type: ndcg_at_3
      value: 42.768
    - type: ndcg_at_5
      value: 45.373000000000005
    - type: precision_at_1
      value: 37.536
    - type: precision_at_10
      value: 8.681
    - type: precision_at_100
      value: 1.34
    - type: precision_at_1000
      value: 0.165
    - type: precision_at_3
      value: 20.468
    - type: precision_at_5
      value: 14.495
    - type: recall_at_1
      value: 30.473
    - type: recall_at_10
      value: 60.092999999999996
    - type: recall_at_100
      value: 82.733
    - type: recall_at_1000
      value: 94.875
    - type: recall_at_3
      value: 45.734
    - type: recall_at_5
      value: 52.691
    - type: map_at_1
      value: 29.976000000000003
    - type: map_at_10
      value: 41.097
    - type: map_at_100
      value: 42.547000000000004
    - type: map_at_1000
      value: 42.659000000000006
    - type: map_at_3
      value: 37.251
    - type: map_at_5
      value: 39.493
    - type: mrr_at_1
      value: 37.557
    - type: mrr_at_10
      value: 46.605000000000004
    - type: mrr_at_100
      value: 47.487
    - type: mrr_at_1000
      value: 47.54
    - type: mrr_at_3
      value: 43.721
    - type: mrr_at_5
      value: 45.411
    - type: ndcg_at_1
      value: 37.557
    - type: ndcg_at_10
      value: 47.449000000000005
    - type: ndcg_at_100
      value: 53.052
    - type: ndcg_at_1000
      value: 55.010999999999996
    - type: ndcg_at_3
      value: 41.439
    - type: ndcg_at_5
      value: 44.292
    - type: precision_at_1
      value: 37.557
    - type: precision_at_10
      value: 8.847
    - type: precision_at_100
      value: 1.357
    - type: precision_at_1000
      value: 0.16999999999999998
    - type: precision_at_3
      value: 20.091
    - type: precision_at_5
      value: 14.384
    - type: recall_at_1
      value: 29.976000000000003
    - type: recall_at_10
      value: 60.99099999999999
    - type: recall_at_100
      value: 84.245
    - type: recall_at_1000
      value: 96.97200000000001
    - type: recall_at_3
      value: 43.794
    - type: recall_at_5
      value: 51.778999999999996
    - type: map_at_1
      value: 28.099166666666665
    - type: map_at_10
      value: 38.1365
    - type: map_at_100
      value: 39.44491666666667
    - type: map_at_1000
      value: 39.55858333333334
    - type: map_at_3
      value: 35.03641666666666
    - type: map_at_5
      value: 36.79833333333334
    - type: mrr_at_1
      value: 33.39966666666667
    - type: mrr_at_10
      value: 42.42583333333333
    - type: mrr_at_100
      value: 43.28575
    - type: mrr_at_1000
      value: 43.33741666666667
    - type: mrr_at_3
      value: 39.94975
    - type: mrr_at_5
      value: 41.41633333333334
    - type: ndcg_at_1
      value: 33.39966666666667
    - type: ndcg_at_10
      value: 43.81741666666667
    - type: ndcg_at_100
      value: 49.08166666666667
    - type: ndcg_at_1000
      value: 51.121166666666674
    - type: ndcg_at_3
      value: 38.73575
    - type: ndcg_at_5
      value: 41.18158333333333
    - type: precision_at_1
      value: 33.39966666666667
    - type: precision_at_10
      value: 7.738916666666667
    - type: precision_at_100
      value: 1.2265833333333331
    - type: precision_at_1000
      value: 0.15983333333333336
    - type: precision_at_3
      value: 17.967416666666665
    - type: precision_at_5
      value: 12.78675
    - type: recall_at_1
      value: 28.099166666666665
    - type: recall_at_10
      value: 56.27049999999999
    - type: recall_at_100
      value: 78.93291666666667
    - type: recall_at_1000
      value: 92.81608333333334
    - type: recall_at_3
      value: 42.09775
    - type: recall_at_5
      value: 48.42533333333334
    - type: map_at_1
      value: 23.663
    - type: map_at_10
      value: 30.377
    - type: map_at_100
      value: 31.426
    - type: map_at_1000
      value: 31.519000000000002
    - type: map_at_3
      value: 28.069
    - type: map_at_5
      value: 29.256999999999998
    - type: mrr_at_1
      value: 26.687
    - type: mrr_at_10
      value: 33.107
    - type: mrr_at_100
      value: 34.055
    - type: mrr_at_1000
      value: 34.117999999999995
    - type: mrr_at_3
      value: 31.058000000000003
    - type: mrr_at_5
      value: 32.14
    - type: ndcg_at_1
      value: 26.687
    - type: ndcg_at_10
      value: 34.615
    - type: ndcg_at_100
      value: 39.776
    - type: ndcg_at_1000
      value: 42.05
    - type: ndcg_at_3
      value: 30.322
    - type: ndcg_at_5
      value: 32.157000000000004
    - type: precision_at_1
      value: 26.687
    - type: precision_at_10
      value: 5.491
    - type: precision_at_100
      value: 0.877
    - type: precision_at_1000
      value: 0.11499999999999999
    - type: precision_at_3
      value: 13.139000000000001
    - type: precision_at_5
      value: 9.049
    - type: recall_at_1
      value: 23.663
    - type: recall_at_10
      value: 45.035
    - type: recall_at_100
      value: 68.554
    - type: recall_at_1000
      value: 85.077
    - type: recall_at_3
      value: 32.982
    - type: recall_at_5
      value: 37.688
    - type: map_at_1
      value: 17.403
    - type: map_at_10
      value: 25.197000000000003
    - type: map_at_100
      value: 26.355
    - type: map_at_1000
      value: 26.487
    - type: map_at_3
      value: 22.733
    - type: map_at_5
      value: 24.114
    - type: mrr_at_1
      value: 21.37
    - type: mrr_at_10
      value: 29.091
    - type: mrr_at_100
      value: 30.018
    - type: mrr_at_1000
      value: 30.096
    - type: mrr_at_3
      value: 26.887
    - type: mrr_at_5
      value: 28.157
    - type: ndcg_at_1
      value: 21.37
    - type: ndcg_at_10
      value: 30.026000000000003
    - type: ndcg_at_100
      value: 35.416
    - type: ndcg_at_1000
      value: 38.45
    - type: ndcg_at_3
      value: 25.764
    - type: ndcg_at_5
      value: 27.742
    - type: precision_at_1
      value: 21.37
    - type: precision_at_10
      value: 5.609
    - type: precision_at_100
      value: 0.9860000000000001
    - type: precision_at_1000
      value: 0.14300000000000002
    - type: precision_at_3
      value: 12.423
    - type: precision_at_5
      value: 9.009
    - type: recall_at_1
      value: 17.403
    - type: recall_at_10
      value: 40.573
    - type: recall_at_100
      value: 64.818
    - type: recall_at_1000
      value: 86.53699999999999
    - type: recall_at_3
      value: 28.493000000000002
    - type: recall_at_5
      value: 33.660000000000004
    - type: map_at_1
      value: 28.639
    - type: map_at_10
      value: 38.951
    - type: map_at_100
      value: 40.238
    - type: map_at_1000
      value: 40.327
    - type: map_at_3
      value: 35.842
    - type: map_at_5
      value: 37.617
    - type: mrr_at_1
      value: 33.769
    - type: mrr_at_10
      value: 43.088
    - type: mrr_at_100
      value: 44.03
    - type: mrr_at_1000
      value: 44.072
    - type: mrr_at_3
      value: 40.656
    - type: mrr_at_5
      value: 42.138999999999996
    - type: ndcg_at_1
      value: 33.769
    - type: ndcg_at_10
      value: 44.676
    - type: ndcg_at_100
      value: 50.416000000000004
    - type: ndcg_at_1000
      value: 52.227999999999994
    - type: ndcg_at_3
      value: 39.494
    - type: ndcg_at_5
      value: 42.013
    - type: precision_at_1
      value: 33.769
    - type: precision_at_10
      value: 7.668
    - type: precision_at_100
      value: 1.18
    - type: precision_at_1000
      value: 0.145
    - type: precision_at_3
      value: 18.221
    - type: precision_at_5
      value: 12.966
    - type: recall_at_1
      value: 28.639
    - type: recall_at_10
      value: 57.687999999999995
    - type: recall_at_100
      value: 82.541
    - type: recall_at_1000
      value: 94.896
    - type: recall_at_3
      value: 43.651
    - type: recall_at_5
      value: 49.925999999999995
    - type: map_at_1
      value: 29.57
    - type: map_at_10
      value: 40.004
    - type: map_at_100
      value: 41.75
    - type: map_at_1000
      value: 41.97
    - type: map_at_3
      value: 36.788
    - type: map_at_5
      value: 38.671
    - type: mrr_at_1
      value: 35.375
    - type: mrr_at_10
      value: 45.121
    - type: mrr_at_100
      value: 45.994
    - type: mrr_at_1000
      value: 46.04
    - type: mrr_at_3
      value: 42.227
    - type: mrr_at_5
      value: 43.995
    - type: ndcg_at_1
      value: 35.375
    - type: ndcg_at_10
      value: 46.392
    - type: ndcg_at_100
      value: 52.196
    - type: ndcg_at_1000
      value: 54.274
    - type: ndcg_at_3
      value: 41.163
    - type: ndcg_at_5
      value: 43.813
    - type: precision_at_1
      value: 35.375
    - type: precision_at_10
      value: 8.676
    - type: precision_at_100
      value: 1.678
    - type: precision_at_1000
      value: 0.253
    - type: precision_at_3
      value: 19.104
    - type: precision_at_5
      value: 13.913
    - type: recall_at_1
      value: 29.57
    - type: recall_at_10
      value: 58.779
    - type: recall_at_100
      value: 83.337
    - type: recall_at_1000
      value: 95.979
    - type: recall_at_3
      value: 44.005
    - type: recall_at_5
      value: 50.975
    - type: map_at_1
      value: 20.832
    - type: map_at_10
      value: 29.733999999999998
    - type: map_at_100
      value: 30.727
    - type: map_at_1000
      value: 30.843999999999998
    - type: map_at_3
      value: 26.834999999999997
    - type: map_at_5
      value: 28.555999999999997
    - type: mrr_at_1
      value: 22.921
    - type: mrr_at_10
      value: 31.791999999999998
    - type: mrr_at_100
      value: 32.666000000000004
    - type: mrr_at_1000
      value: 32.751999999999995
    - type: mrr_at_3
      value: 29.144
    - type: mrr_at_5
      value: 30.622
    - type: ndcg_at_1
      value: 22.921
    - type: ndcg_at_10
      value: 34.915
    - type: ndcg_at_100
      value: 39.744
    - type: ndcg_at_1000
      value: 42.407000000000004
    - type: ndcg_at_3
      value: 29.421000000000003
    - type: ndcg_at_5
      value: 32.211
    - type: precision_at_1
      value: 22.921
    - type: precision_at_10
      value: 5.675
    - type: precision_at_100
      value: 0.872
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 12.753999999999998
    - type: precision_at_5
      value: 9.353
    - type: recall_at_1
      value: 20.832
    - type: recall_at_10
      value: 48.795
    - type: recall_at_100
      value: 70.703
    - type: recall_at_1000
      value: 90.187
    - type: recall_at_3
      value: 34.455000000000005
    - type: recall_at_5
      value: 40.967
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
      value: 10.334
    - type: map_at_10
      value: 19.009999999999998
    - type: map_at_100
      value: 21.129
    - type: map_at_1000
      value: 21.328
    - type: map_at_3
      value: 15.152
    - type: map_at_5
      value: 17.084
    - type: mrr_at_1
      value: 23.453
    - type: mrr_at_10
      value: 36.099
    - type: mrr_at_100
      value: 37.069
    - type: mrr_at_1000
      value: 37.104
    - type: mrr_at_3
      value: 32.096000000000004
    - type: mrr_at_5
      value: 34.451
    - type: ndcg_at_1
      value: 23.453
    - type: ndcg_at_10
      value: 27.739000000000004
    - type: ndcg_at_100
      value: 35.836
    - type: ndcg_at_1000
      value: 39.242
    - type: ndcg_at_3
      value: 21.263
    - type: ndcg_at_5
      value: 23.677
    - type: precision_at_1
      value: 23.453
    - type: precision_at_10
      value: 9.199
    - type: precision_at_100
      value: 1.791
    - type: precision_at_1000
      value: 0.242
    - type: precision_at_3
      value: 16.2
    - type: precision_at_5
      value: 13.147
    - type: recall_at_1
      value: 10.334
    - type: recall_at_10
      value: 35.177
    - type: recall_at_100
      value: 63.009
    - type: recall_at_1000
      value: 81.938
    - type: recall_at_3
      value: 19.914
    - type: recall_at_5
      value: 26.077
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
      value: 8.212
    - type: map_at_10
      value: 17.386
    - type: map_at_100
      value: 24.234
    - type: map_at_1000
      value: 25.724999999999998
    - type: map_at_3
      value: 12.727
    - type: map_at_5
      value: 14.785
    - type: mrr_at_1
      value: 59.25
    - type: mrr_at_10
      value: 68.687
    - type: mrr_at_100
      value: 69.133
    - type: mrr_at_1000
      value: 69.14099999999999
    - type: mrr_at_3
      value: 66.917
    - type: mrr_at_5
      value: 67.742
    - type: ndcg_at_1
      value: 48.625
    - type: ndcg_at_10
      value: 36.675999999999995
    - type: ndcg_at_100
      value: 41.543
    - type: ndcg_at_1000
      value: 49.241
    - type: ndcg_at_3
      value: 41.373
    - type: ndcg_at_5
      value: 38.707
    - type: precision_at_1
      value: 59.25
    - type: precision_at_10
      value: 28.525
    - type: precision_at_100
      value: 9.027000000000001
    - type: precision_at_1000
      value: 1.8339999999999999
    - type: precision_at_3
      value: 44.833
    - type: precision_at_5
      value: 37.35
    - type: recall_at_1
      value: 8.212
    - type: recall_at_10
      value: 23.188
    - type: recall_at_100
      value: 48.613
    - type: recall_at_1000
      value: 73.093
    - type: recall_at_3
      value: 14.419
    - type: recall_at_5
      value: 17.798
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
      value: 52.725
    - type: f1
      value: 46.50743309855908
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
      value: 55.086
    - type: map_at_10
      value: 66.914
    - type: map_at_100
      value: 67.321
    - type: map_at_1000
      value: 67.341
    - type: map_at_3
      value: 64.75800000000001
    - type: map_at_5
      value: 66.189
    - type: mrr_at_1
      value: 59.28600000000001
    - type: mrr_at_10
      value: 71.005
    - type: mrr_at_100
      value: 71.304
    - type: mrr_at_1000
      value: 71.313
    - type: mrr_at_3
      value: 69.037
    - type: mrr_at_5
      value: 70.35
    - type: ndcg_at_1
      value: 59.28600000000001
    - type: ndcg_at_10
      value: 72.695
    - type: ndcg_at_100
      value: 74.432
    - type: ndcg_at_1000
      value: 74.868
    - type: ndcg_at_3
      value: 68.72200000000001
    - type: ndcg_at_5
      value: 71.081
    - type: precision_at_1
      value: 59.28600000000001
    - type: precision_at_10
      value: 9.499
    - type: precision_at_100
      value: 1.052
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 27.503
    - type: precision_at_5
      value: 17.854999999999997
    - type: recall_at_1
      value: 55.086
    - type: recall_at_10
      value: 86.453
    - type: recall_at_100
      value: 94.028
    - type: recall_at_1000
      value: 97.052
    - type: recall_at_3
      value: 75.821
    - type: recall_at_5
      value: 81.6
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
      value: 22.262999999999998
    - type: map_at_10
      value: 37.488
    - type: map_at_100
      value: 39.498
    - type: map_at_1000
      value: 39.687
    - type: map_at_3
      value: 32.529
    - type: map_at_5
      value: 35.455
    - type: mrr_at_1
      value: 44.907000000000004
    - type: mrr_at_10
      value: 53.239000000000004
    - type: mrr_at_100
      value: 54.086
    - type: mrr_at_1000
      value: 54.122
    - type: mrr_at_3
      value: 51.235
    - type: mrr_at_5
      value: 52.415
    - type: ndcg_at_1
      value: 44.907000000000004
    - type: ndcg_at_10
      value: 45.446
    - type: ndcg_at_100
      value: 52.429
    - type: ndcg_at_1000
      value: 55.169000000000004
    - type: ndcg_at_3
      value: 41.882000000000005
    - type: ndcg_at_5
      value: 43.178
    - type: precision_at_1
      value: 44.907000000000004
    - type: precision_at_10
      value: 12.931999999999999
    - type: precision_at_100
      value: 2.025
    - type: precision_at_1000
      value: 0.248
    - type: precision_at_3
      value: 28.652
    - type: precision_at_5
      value: 21.204
    - type: recall_at_1
      value: 22.262999999999998
    - type: recall_at_10
      value: 52.447
    - type: recall_at_100
      value: 78.045
    - type: recall_at_1000
      value: 94.419
    - type: recall_at_3
      value: 38.064
    - type: recall_at_5
      value: 44.769
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
      value: 32.519
    - type: map_at_10
      value: 45.831
    - type: map_at_100
      value: 46.815
    - type: map_at_1000
      value: 46.899
    - type: map_at_3
      value: 42.836
    - type: map_at_5
      value: 44.65
    - type: mrr_at_1
      value: 65.037
    - type: mrr_at_10
      value: 72.16
    - type: mrr_at_100
      value: 72.51100000000001
    - type: mrr_at_1000
      value: 72.53
    - type: mrr_at_3
      value: 70.682
    - type: mrr_at_5
      value: 71.54599999999999
    - type: ndcg_at_1
      value: 65.037
    - type: ndcg_at_10
      value: 55.17999999999999
    - type: ndcg_at_100
      value: 58.888
    - type: ndcg_at_1000
      value: 60.648
    - type: ndcg_at_3
      value: 50.501
    - type: ndcg_at_5
      value: 52.977
    - type: precision_at_1
      value: 65.037
    - type: precision_at_10
      value: 11.530999999999999
    - type: precision_at_100
      value: 1.4460000000000002
    - type: precision_at_1000
      value: 0.168
    - type: precision_at_3
      value: 31.483
    - type: precision_at_5
      value: 20.845
    - type: recall_at_1
      value: 32.519
    - type: recall_at_10
      value: 57.657000000000004
    - type: recall_at_100
      value: 72.30199999999999
    - type: recall_at_1000
      value: 84.024
    - type: recall_at_3
      value: 47.225
    - type: recall_at_5
      value: 52.113
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
      value: 88.3168
    - type: ap
      value: 83.80165516037135
    - type: f1
      value: 88.29942471066407
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
      value: 20.724999999999998
    - type: map_at_10
      value: 32.736
    - type: map_at_100
      value: 33.938
    - type: map_at_1000
      value: 33.991
    - type: map_at_3
      value: 28.788000000000004
    - type: map_at_5
      value: 31.016
    - type: mrr_at_1
      value: 21.361
    - type: mrr_at_10
      value: 33.323
    - type: mrr_at_100
      value: 34.471000000000004
    - type: mrr_at_1000
      value: 34.518
    - type: mrr_at_3
      value: 29.453000000000003
    - type: mrr_at_5
      value: 31.629
    - type: ndcg_at_1
      value: 21.361
    - type: ndcg_at_10
      value: 39.649
    - type: ndcg_at_100
      value: 45.481
    - type: ndcg_at_1000
      value: 46.775
    - type: ndcg_at_3
      value: 31.594
    - type: ndcg_at_5
      value: 35.543
    - type: precision_at_1
      value: 21.361
    - type: precision_at_10
      value: 6.3740000000000006
    - type: precision_at_100
      value: 0.931
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 13.514999999999999
    - type: precision_at_5
      value: 10.100000000000001
    - type: recall_at_1
      value: 20.724999999999998
    - type: recall_at_10
      value: 61.034
    - type: recall_at_100
      value: 88.062
    - type: recall_at_1000
      value: 97.86399999999999
    - type: recall_at_3
      value: 39.072
    - type: recall_at_5
      value: 48.53
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
      value: 93.8919288645691
    - type: f1
      value: 93.57059586398059
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
      value: 67.97993616051072
    - type: f1
      value: 48.244319183606535
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
      value: 68.90047074646941
    - type: f1
      value: 66.48999056063725
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
      value: 73.34566240753195
    - type: f1
      value: 73.54164154290658
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
      value: 34.21866934757011
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
      value: 32.000936217235534
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
      value: 31.68189362520352
    - type: mrr
      value: 32.69603637784303
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
      value: 6.078
    - type: map_at_10
      value: 12.671
    - type: map_at_100
      value: 16.291
    - type: map_at_1000
      value: 17.855999999999998
    - type: map_at_3
      value: 9.610000000000001
    - type: map_at_5
      value: 11.152
    - type: mrr_at_1
      value: 43.963
    - type: mrr_at_10
      value: 53.173
    - type: mrr_at_100
      value: 53.718999999999994
    - type: mrr_at_1000
      value: 53.756
    - type: mrr_at_3
      value: 50.980000000000004
    - type: mrr_at_5
      value: 52.42
    - type: ndcg_at_1
      value: 42.415000000000006
    - type: ndcg_at_10
      value: 34.086
    - type: ndcg_at_100
      value: 32.545
    - type: ndcg_at_1000
      value: 41.144999999999996
    - type: ndcg_at_3
      value: 39.434999999999995
    - type: ndcg_at_5
      value: 37.888
    - type: precision_at_1
      value: 43.653
    - type: precision_at_10
      value: 25.014999999999997
    - type: precision_at_100
      value: 8.594
    - type: precision_at_1000
      value: 2.169
    - type: precision_at_3
      value: 37.049
    - type: precision_at_5
      value: 33.065
    - type: recall_at_1
      value: 6.078
    - type: recall_at_10
      value: 16.17
    - type: recall_at_100
      value: 34.512
    - type: recall_at_1000
      value: 65.447
    - type: recall_at_3
      value: 10.706
    - type: recall_at_5
      value: 13.158
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
      value: 27.378000000000004
    - type: map_at_10
      value: 42.178
    - type: map_at_100
      value: 43.32
    - type: map_at_1000
      value: 43.358000000000004
    - type: map_at_3
      value: 37.474000000000004
    - type: map_at_5
      value: 40.333000000000006
    - type: mrr_at_1
      value: 30.823
    - type: mrr_at_10
      value: 44.626
    - type: mrr_at_100
      value: 45.494
    - type: mrr_at_1000
      value: 45.519
    - type: mrr_at_3
      value: 40.585
    - type: mrr_at_5
      value: 43.146
    - type: ndcg_at_1
      value: 30.794
    - type: ndcg_at_10
      value: 50.099000000000004
    - type: ndcg_at_100
      value: 54.900999999999996
    - type: ndcg_at_1000
      value: 55.69499999999999
    - type: ndcg_at_3
      value: 41.238
    - type: ndcg_at_5
      value: 46.081
    - type: precision_at_1
      value: 30.794
    - type: precision_at_10
      value: 8.549
    - type: precision_at_100
      value: 1.124
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 18.926000000000002
    - type: precision_at_5
      value: 14.16
    - type: recall_at_1
      value: 27.378000000000004
    - type: recall_at_10
      value: 71.842
    - type: recall_at_100
      value: 92.565
    - type: recall_at_1000
      value: 98.402
    - type: recall_at_3
      value: 49.053999999999995
    - type: recall_at_5
      value: 60.207
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
      value: 70.557
    - type: map_at_10
      value: 84.729
    - type: map_at_100
      value: 85.369
    - type: map_at_1000
      value: 85.382
    - type: map_at_3
      value: 81.72
    - type: map_at_5
      value: 83.613
    - type: mrr_at_1
      value: 81.3
    - type: mrr_at_10
      value: 87.488
    - type: mrr_at_100
      value: 87.588
    - type: mrr_at_1000
      value: 87.589
    - type: mrr_at_3
      value: 86.53
    - type: mrr_at_5
      value: 87.18599999999999
    - type: ndcg_at_1
      value: 81.28999999999999
    - type: ndcg_at_10
      value: 88.442
    - type: ndcg_at_100
      value: 89.637
    - type: ndcg_at_1000
      value: 89.70700000000001
    - type: ndcg_at_3
      value: 85.55199999999999
    - type: ndcg_at_5
      value: 87.154
    - type: precision_at_1
      value: 81.28999999999999
    - type: precision_at_10
      value: 13.489999999999998
    - type: precision_at_100
      value: 1.54
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.553
    - type: precision_at_5
      value: 24.708
    - type: recall_at_1
      value: 70.557
    - type: recall_at_10
      value: 95.645
    - type: recall_at_100
      value: 99.693
    - type: recall_at_1000
      value: 99.995
    - type: recall_at_3
      value: 87.359
    - type: recall_at_5
      value: 91.89699999999999
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
      value: 63.65060114776209
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
      value: 64.63271250680617
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
      value: 4.263
    - type: map_at_10
      value: 10.801
    - type: map_at_100
      value: 12.888
    - type: map_at_1000
      value: 13.224
    - type: map_at_3
      value: 7.362
    - type: map_at_5
      value: 9.149000000000001
    - type: mrr_at_1
      value: 21
    - type: mrr_at_10
      value: 31.416
    - type: mrr_at_100
      value: 32.513
    - type: mrr_at_1000
      value: 32.58
    - type: mrr_at_3
      value: 28.116999999999997
    - type: mrr_at_5
      value: 29.976999999999997
    - type: ndcg_at_1
      value: 21
    - type: ndcg_at_10
      value: 18.551000000000002
    - type: ndcg_at_100
      value: 26.657999999999998
    - type: ndcg_at_1000
      value: 32.485
    - type: ndcg_at_3
      value: 16.834
    - type: ndcg_at_5
      value: 15.204999999999998
    - type: precision_at_1
      value: 21
    - type: precision_at_10
      value: 9.84
    - type: precision_at_100
      value: 2.16
    - type: precision_at_1000
      value: 0.35500000000000004
    - type: precision_at_3
      value: 15.667
    - type: precision_at_5
      value: 13.62
    - type: recall_at_1
      value: 4.263
    - type: recall_at_10
      value: 19.922
    - type: recall_at_100
      value: 43.808
    - type: recall_at_1000
      value: 72.14500000000001
    - type: recall_at_3
      value: 9.493
    - type: recall_at_5
      value: 13.767999999999999
  - task:
      type: STS
    dataset:
      name: MTEB SICK-R
      type: mteb/sickr-sts
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_spearman
      value: 81.27446313317233
  - task:
      type: STS
    dataset:
      name: MTEB STS12
      type: mteb/sts12-sts
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_spearman
      value: 76.27963301217527
  - task:
      type: STS
    dataset:
      name: MTEB STS13
      type: mteb/sts13-sts
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_spearman
      value: 88.18495048450949
  - task:
      type: STS
    dataset:
      name: MTEB STS14
      type: mteb/sts14-sts
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_spearman
      value: 81.91982338692046
  - task:
      type: STS
    dataset:
      name: MTEB STS15
      type: mteb/sts15-sts
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_spearman
      value: 89.00896818385291
  - task:
      type: STS
    dataset:
      name: MTEB STS16
      type: mteb/sts16-sts
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_spearman
      value: 85.48814644586132
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-en)
      type: mteb/sts17-crosslingual-sts
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_spearman
      value: 90.30116926966582
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (en)
      type: mteb/sts22-crosslingual-sts
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_spearman
      value: 67.74132963032342
  - task:
      type: STS
    dataset:
      name: MTEB STSBenchmark
      type: mteb/stsbenchmark-sts
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_spearman
      value: 86.87741355780479
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
      value: 82.0019012295875
    - type: mrr
      value: 94.70267024188593
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
      value: 50.05
    - type: map_at_10
      value: 59.36
    - type: map_at_100
      value: 59.967999999999996
    - type: map_at_1000
      value: 60.023
    - type: map_at_3
      value: 56.515
    - type: map_at_5
      value: 58.272999999999996
    - type: mrr_at_1
      value: 53
    - type: mrr_at_10
      value: 61.102000000000004
    - type: mrr_at_100
      value: 61.476
    - type: mrr_at_1000
      value: 61.523
    - type: mrr_at_3
      value: 58.778
    - type: mrr_at_5
      value: 60.128
    - type: ndcg_at_1
      value: 53
    - type: ndcg_at_10
      value: 64.43100000000001
    - type: ndcg_at_100
      value: 66.73599999999999
    - type: ndcg_at_1000
      value: 68.027
    - type: ndcg_at_3
      value: 59.279
    - type: ndcg_at_5
      value: 61.888
    - type: precision_at_1
      value: 53
    - type: precision_at_10
      value: 8.767
    - type: precision_at_100
      value: 1.01
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 23.444000000000003
    - type: precision_at_5
      value: 15.667
    - type: recall_at_1
      value: 50.05
    - type: recall_at_10
      value: 78.511
    - type: recall_at_100
      value: 88.5
    - type: recall_at_1000
      value: 98.333
    - type: recall_at_3
      value: 64.117
    - type: recall_at_5
      value: 70.867
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
      value: 99.72178217821782
    - type: cos_sim_ap
      value: 93.0728601593541
    - type: cos_sim_f1
      value: 85.6727976766699
    - type: cos_sim_precision
      value: 83.02063789868667
    - type: cos_sim_recall
      value: 88.5
    - type: dot_accuracy
      value: 99.72178217821782
    - type: dot_ap
      value: 93.07287396168348
    - type: dot_f1
      value: 85.6727976766699
    - type: dot_precision
      value: 83.02063789868667
    - type: dot_recall
      value: 88.5
    - type: euclidean_accuracy
      value: 99.72178217821782
    - type: euclidean_ap
      value: 93.07285657982895
    - type: euclidean_f1
      value: 85.6727976766699
    - type: euclidean_precision
      value: 83.02063789868667
    - type: euclidean_recall
      value: 88.5
    - type: manhattan_accuracy
      value: 99.72475247524753
    - type: manhattan_ap
      value: 93.02792973059809
    - type: manhattan_f1
      value: 85.7727737973388
    - type: manhattan_precision
      value: 87.84067085953879
    - type: manhattan_recall
      value: 83.8
    - type: max_accuracy
      value: 99.72475247524753
    - type: max_ap
      value: 93.07287396168348
    - type: max_f1
      value: 85.7727737973388
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
      value: 68.77583615550819
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
      value: 36.151636938606956
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
      value: 52.16607939471187
    - type: mrr
      value: 52.95172046091163
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
      value: 31.314646669495666
    - type: cos_sim_spearman
      value: 31.83562491439455
    - type: dot_pearson
      value: 31.314590842874157
    - type: dot_spearman
      value: 31.83363065810437
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
      value: 0.198
    - type: map_at_10
      value: 1.3010000000000002
    - type: map_at_100
      value: 7.2139999999999995
    - type: map_at_1000
      value: 20.179
    - type: map_at_3
      value: 0.528
    - type: map_at_5
      value: 0.8019999999999999
    - type: mrr_at_1
      value: 72
    - type: mrr_at_10
      value: 83.39999999999999
    - type: mrr_at_100
      value: 83.39999999999999
    - type: mrr_at_1000
      value: 83.39999999999999
    - type: mrr_at_3
      value: 81.667
    - type: mrr_at_5
      value: 83.06700000000001
    - type: ndcg_at_1
      value: 66
    - type: ndcg_at_10
      value: 58.059000000000005
    - type: ndcg_at_100
      value: 44.316
    - type: ndcg_at_1000
      value: 43.147000000000006
    - type: ndcg_at_3
      value: 63.815999999999995
    - type: ndcg_at_5
      value: 63.005
    - type: precision_at_1
      value: 72
    - type: precision_at_10
      value: 61.4
    - type: precision_at_100
      value: 45.62
    - type: precision_at_1000
      value: 19.866
    - type: precision_at_3
      value: 70
    - type: precision_at_5
      value: 68.8
    - type: recall_at_1
      value: 0.198
    - type: recall_at_10
      value: 1.517
    - type: recall_at_100
      value: 10.587
    - type: recall_at_1000
      value: 41.233
    - type: recall_at_3
      value: 0.573
    - type: recall_at_5
      value: 0.907
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
      value: 1.894
    - type: map_at_10
      value: 8.488999999999999
    - type: map_at_100
      value: 14.445
    - type: map_at_1000
      value: 16.078
    - type: map_at_3
      value: 4.589
    - type: map_at_5
      value: 6.019
    - type: mrr_at_1
      value: 22.448999999999998
    - type: mrr_at_10
      value: 39.82
    - type: mrr_at_100
      value: 40.752
    - type: mrr_at_1000
      value: 40.771
    - type: mrr_at_3
      value: 34.354
    - type: mrr_at_5
      value: 37.721
    - type: ndcg_at_1
      value: 19.387999999999998
    - type: ndcg_at_10
      value: 21.563
    - type: ndcg_at_100
      value: 33.857
    - type: ndcg_at_1000
      value: 46.199
    - type: ndcg_at_3
      value: 22.296
    - type: ndcg_at_5
      value: 21.770999999999997
    - type: precision_at_1
      value: 22.448999999999998
    - type: precision_at_10
      value: 19.796
    - type: precision_at_100
      value: 7.142999999999999
    - type: precision_at_1000
      value: 1.541
    - type: precision_at_3
      value: 24.490000000000002
    - type: precision_at_5
      value: 22.448999999999998
    - type: recall_at_1
      value: 1.894
    - type: recall_at_10
      value: 14.931
    - type: recall_at_100
      value: 45.524
    - type: recall_at_1000
      value: 83.243
    - type: recall_at_3
      value: 5.712
    - type: recall_at_5
      value: 8.386000000000001
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
      value: 71.049
    - type: ap
      value: 13.85116971310922
    - type: f1
      value: 54.37504302487686
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
      value: 64.1312959818902
    - type: f1
      value: 64.11413877009383
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
      value: 54.13103431861502
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
      value: 87.327889372355
    - type: cos_sim_ap
      value: 77.42059895975699
    - type: cos_sim_f1
      value: 71.02706903250873
    - type: cos_sim_precision
      value: 69.75324344950394
    - type: cos_sim_recall
      value: 72.34828496042216
    - type: dot_accuracy
      value: 87.327889372355
    - type: dot_ap
      value: 77.4209479346677
    - type: dot_f1
      value: 71.02706903250873
    - type: dot_precision
      value: 69.75324344950394
    - type: dot_recall
      value: 72.34828496042216
    - type: euclidean_accuracy
      value: 87.327889372355
    - type: euclidean_ap
      value: 77.42096495861037
    - type: euclidean_f1
      value: 71.02706903250873
    - type: euclidean_precision
      value: 69.75324344950394
    - type: euclidean_recall
      value: 72.34828496042216
    - type: manhattan_accuracy
      value: 87.31000774870358
    - type: manhattan_ap
      value: 77.38930750711619
    - type: manhattan_f1
      value: 71.07935314027831
    - type: manhattan_precision
      value: 67.70957726295677
    - type: manhattan_recall
      value: 74.80211081794195
    - type: max_accuracy
      value: 87.327889372355
    - type: max_ap
      value: 77.42096495861037
    - type: max_f1
      value: 71.07935314027831
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
      value: 89.58939729110878
    - type: cos_sim_ap
      value: 87.17594155025475
    - type: cos_sim_f1
      value: 79.21146953405018
    - type: cos_sim_precision
      value: 76.8918527109307
    - type: cos_sim_recall
      value: 81.67539267015707
    - type: dot_accuracy
      value: 89.58939729110878
    - type: dot_ap
      value: 87.17593963273593
    - type: dot_f1
      value: 79.21146953405018
    - type: dot_precision
      value: 76.8918527109307
    - type: dot_recall
      value: 81.67539267015707
    - type: euclidean_accuracy
      value: 89.58939729110878
    - type: euclidean_ap
      value: 87.17592466925834
    - type: euclidean_f1
      value: 79.21146953405018
    - type: euclidean_precision
      value: 76.8918527109307
    - type: euclidean_recall
      value: 81.67539267015707
    - type: manhattan_accuracy
      value: 89.62626615438352
    - type: manhattan_ap
      value: 87.16589873161546
    - type: manhattan_f1
      value: 79.25143598295348
    - type: manhattan_precision
      value: 76.39494177323712
    - type: manhattan_recall
      value: 82.32984293193716
    - type: max_accuracy
      value: 89.62626615438352
    - type: max_ap
      value: 87.17594155025475
    - type: max_f1
      value: 79.25143598295348
---

# hkunlp/instructor-large
We introduce **Instructor**👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) ***by simply providing the task instruction, without any finetuning***. Instructor👨‍ achieves sota on 70 diverse embedding tasks ([MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard))!
The model is easy to use with **our customized** `sentence-transformer` library. For more details, check out [our paper](https://arxiv.org/abs/2212.09741) and [project page](https://instructor-embedding.github.io/)! 

**************************** **Updates** ****************************

* 12/28: We released a new [checkpoint](https://huggingface.co/hkunlp/instructor-large) trained with hard negatives, which gives better performance.
* 12/21: We released our [paper](https://arxiv.org/abs/2212.09741), [code](https://github.com/HKUNLP/instructor-embedding), [checkpoint](https://huggingface.co/hkunlp/instructor-large) and [project page](https://instructor-embedding.github.io/)! Check them out!

## Quick start
<hr />

## Installation
```bash
pip install InstructorEmbedding
```

## Compute your customized embeddings
Then you can use the model like this to calculate domain-specific and task-aware embeddings:
```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction,sentence]])
print(embeddings)
```

## Use cases
<hr />

## Calculate embeddings for your customized texts
If you want to calculate customized embeddings for specific sentences, you may follow the unified template to write instructions: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Represent the `domain` `text_type` for `task_objective`:
* `domain` is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
* `text_type` is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
* `task_objective` is optional, and it specifies the objective of embedding, e.g., retrieve a document, classify the sentence, etc.

## Calculate Sentence similarities
You can further use the model to compute similarities between two groups of sentences, with **customized embeddings**.
```python
from sklearn.metrics.pairwise import cosine_similarity
sentences_a = [['Represent the Science sentence: ','Parton energy loss in QCD matter'], 
               ['Represent the Financial statement: ','The Federal Reserve on Wednesday raised its benchmark interest rate.']]
sentences_b = [['Represent the Science sentence: ','The Chiral Phase Transition in Dissipative Dynamics'],
               ['Represent the Financial statement: ','The funds rose less than 0.5 per cent on Friday']]
embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities = cosine_similarity(embeddings_a,embeddings_b)
print(similarities)
```

## Information Retrieval
You can also use **customized embeddings** for information retrieval.
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
query  = [['Represent the Wikipedia question for retrieving supporting documents: ','where is the food stored in a yam plant']]
corpus = [['Represent the Wikipedia document for retrieval: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
          ['Represent the Wikipedia document for retrieval: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
          ['Represent the Wikipedia document for retrieval: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.']]
query_embeddings = model.encode(query)
corpus_embeddings = model.encode(corpus)
similarities = cosine_similarity(query_embeddings,corpus_embeddings)
retrieved_doc_id = np.argmax(similarities)
print(retrieved_doc_id)
```

## Clustering
Use **customized embeddings** for clustering texts in groups.
```python
import sklearn.cluster
sentences = [['Represent the Medicine sentence for clustering: ','Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity'],
             ['Represent the Medicine sentence for clustering: ','Comparison of Atmospheric Neutrino Flux Calculations at Low Energies'],
             ['Represent the Medicine sentence for clustering: ','Fermion Bags in the Massive Gross-Neveu Model'],
             ['Represent the Medicine sentence for clustering: ',"QCD corrections to Associated t-tbar-H production at the Tevatron"],
             ['Represent the Medicine sentence for clustering: ','A New Analysis of the R Measurements: Resonance Parameters of the Higher,  Vector States of Charmonium']]
embeddings = model.encode(sentences)
clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=2)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_
print(cluster_assignment)
```