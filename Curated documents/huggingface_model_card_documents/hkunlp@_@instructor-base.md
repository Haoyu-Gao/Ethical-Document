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
- name: final_base_results
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
      value: 86.2089552238806
    - type: ap
      value: 55.76273850794966
    - type: f1
      value: 81.26104211414781
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
      value: 88.35995000000001
    - type: ap
      value: 84.18839957309655
    - type: f1
      value: 88.317619250081
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
      value: 44.64
    - type: f1
      value: 42.48663956478136
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
      value: 27.383000000000003
    - type: map_at_10
      value: 43.024
    - type: map_at_100
      value: 44.023
    - type: map_at_1000
      value: 44.025999999999996
    - type: map_at_3
      value: 37.684
    - type: map_at_5
      value: 40.884
    - type: mrr_at_1
      value: 28.094
    - type: mrr_at_10
      value: 43.315
    - type: mrr_at_100
      value: 44.313
    - type: mrr_at_1000
      value: 44.317
    - type: mrr_at_3
      value: 37.862
    - type: mrr_at_5
      value: 41.155
    - type: ndcg_at_1
      value: 27.383000000000003
    - type: ndcg_at_10
      value: 52.032000000000004
    - type: ndcg_at_100
      value: 56.19499999999999
    - type: ndcg_at_1000
      value: 56.272
    - type: ndcg_at_3
      value: 41.166000000000004
    - type: ndcg_at_5
      value: 46.92
    - type: precision_at_1
      value: 27.383000000000003
    - type: precision_at_10
      value: 8.087
    - type: precision_at_100
      value: 0.989
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 17.093
    - type: precision_at_5
      value: 13.044
    - type: recall_at_1
      value: 27.383000000000003
    - type: recall_at_10
      value: 80.868
    - type: recall_at_100
      value: 98.86200000000001
    - type: recall_at_1000
      value: 99.431
    - type: recall_at_3
      value: 51.28
    - type: recall_at_5
      value: 65.22
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
      value: 39.68441054431849
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
      value: 29.188539728343844
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
      value: 63.173362687519784
    - type: mrr
      value: 76.18860748362133
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
      value: 82.30789953771232
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
      value: 77.03571428571428
    - type: f1
      value: 75.87384305045917
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
      value: 32.98041170516364
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
      value: 25.71652988451154
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
      value: 33.739999999999995
    - type: map_at_10
      value: 46.197
    - type: map_at_100
      value: 47.814
    - type: map_at_1000
      value: 47.934
    - type: map_at_3
      value: 43.091
    - type: map_at_5
      value: 44.81
    - type: mrr_at_1
      value: 41.059
    - type: mrr_at_10
      value: 52.292
    - type: mrr_at_100
      value: 52.978
    - type: mrr_at_1000
      value: 53.015
    - type: mrr_at_3
      value: 49.976
    - type: mrr_at_5
      value: 51.449999999999996
    - type: ndcg_at_1
      value: 41.059
    - type: ndcg_at_10
      value: 52.608
    - type: ndcg_at_100
      value: 57.965
    - type: ndcg_at_1000
      value: 59.775999999999996
    - type: ndcg_at_3
      value: 48.473
    - type: ndcg_at_5
      value: 50.407999999999994
    - type: precision_at_1
      value: 41.059
    - type: precision_at_10
      value: 9.943
    - type: precision_at_100
      value: 1.6070000000000002
    - type: precision_at_1000
      value: 0.20500000000000002
    - type: precision_at_3
      value: 23.413999999999998
    - type: precision_at_5
      value: 16.481
    - type: recall_at_1
      value: 33.739999999999995
    - type: recall_at_10
      value: 63.888999999999996
    - type: recall_at_100
      value: 85.832
    - type: recall_at_1000
      value: 97.475
    - type: recall_at_3
      value: 51.953
    - type: recall_at_5
      value: 57.498000000000005
    - type: map_at_1
      value: 31.169999999999998
    - type: map_at_10
      value: 41.455
    - type: map_at_100
      value: 42.716
    - type: map_at_1000
      value: 42.847
    - type: map_at_3
      value: 38.568999999999996
    - type: map_at_5
      value: 40.099000000000004
    - type: mrr_at_1
      value: 39.427
    - type: mrr_at_10
      value: 47.818
    - type: mrr_at_100
      value: 48.519
    - type: mrr_at_1000
      value: 48.558
    - type: mrr_at_3
      value: 45.86
    - type: mrr_at_5
      value: 46.936
    - type: ndcg_at_1
      value: 39.427
    - type: ndcg_at_10
      value: 47.181
    - type: ndcg_at_100
      value: 51.737
    - type: ndcg_at_1000
      value: 53.74
    - type: ndcg_at_3
      value: 43.261
    - type: ndcg_at_5
      value: 44.891
    - type: precision_at_1
      value: 39.427
    - type: precision_at_10
      value: 8.847
    - type: precision_at_100
      value: 1.425
    - type: precision_at_1000
      value: 0.189
    - type: precision_at_3
      value: 20.785999999999998
    - type: precision_at_5
      value: 14.560999999999998
    - type: recall_at_1
      value: 31.169999999999998
    - type: recall_at_10
      value: 56.971000000000004
    - type: recall_at_100
      value: 76.31400000000001
    - type: recall_at_1000
      value: 88.93900000000001
    - type: recall_at_3
      value: 45.208
    - type: recall_at_5
      value: 49.923
    - type: map_at_1
      value: 39.682
    - type: map_at_10
      value: 52.766000000000005
    - type: map_at_100
      value: 53.84100000000001
    - type: map_at_1000
      value: 53.898
    - type: map_at_3
      value: 49.291000000000004
    - type: map_at_5
      value: 51.365
    - type: mrr_at_1
      value: 45.266
    - type: mrr_at_10
      value: 56.093
    - type: mrr_at_100
      value: 56.763
    - type: mrr_at_1000
      value: 56.793000000000006
    - type: mrr_at_3
      value: 53.668000000000006
    - type: mrr_at_5
      value: 55.1
    - type: ndcg_at_1
      value: 45.266
    - type: ndcg_at_10
      value: 58.836
    - type: ndcg_at_100
      value: 62.863
    - type: ndcg_at_1000
      value: 63.912
    - type: ndcg_at_3
      value: 53.19199999999999
    - type: ndcg_at_5
      value: 56.125
    - type: precision_at_1
      value: 45.266
    - type: precision_at_10
      value: 9.492
    - type: precision_at_100
      value: 1.236
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 23.762
    - type: precision_at_5
      value: 16.414
    - type: recall_at_1
      value: 39.682
    - type: recall_at_10
      value: 73.233
    - type: recall_at_100
      value: 90.335
    - type: recall_at_1000
      value: 97.452
    - type: recall_at_3
      value: 58.562000000000005
    - type: recall_at_5
      value: 65.569
    - type: map_at_1
      value: 26.743
    - type: map_at_10
      value: 34.016000000000005
    - type: map_at_100
      value: 35.028999999999996
    - type: map_at_1000
      value: 35.113
    - type: map_at_3
      value: 31.763
    - type: map_at_5
      value: 33.013999999999996
    - type: mrr_at_1
      value: 28.927000000000003
    - type: mrr_at_10
      value: 36.32
    - type: mrr_at_100
      value: 37.221
    - type: mrr_at_1000
      value: 37.281
    - type: mrr_at_3
      value: 34.105000000000004
    - type: mrr_at_5
      value: 35.371
    - type: ndcg_at_1
      value: 28.927000000000003
    - type: ndcg_at_10
      value: 38.474000000000004
    - type: ndcg_at_100
      value: 43.580000000000005
    - type: ndcg_at_1000
      value: 45.64
    - type: ndcg_at_3
      value: 34.035
    - type: ndcg_at_5
      value: 36.186
    - type: precision_at_1
      value: 28.927000000000003
    - type: precision_at_10
      value: 5.74
    - type: precision_at_100
      value: 0.8710000000000001
    - type: precision_at_1000
      value: 0.108
    - type: precision_at_3
      value: 14.124
    - type: precision_at_5
      value: 9.74
    - type: recall_at_1
      value: 26.743
    - type: recall_at_10
      value: 49.955
    - type: recall_at_100
      value: 73.904
    - type: recall_at_1000
      value: 89.133
    - type: recall_at_3
      value: 38.072
    - type: recall_at_5
      value: 43.266
    - type: map_at_1
      value: 16.928
    - type: map_at_10
      value: 23.549
    - type: map_at_100
      value: 24.887
    - type: map_at_1000
      value: 25.018
    - type: map_at_3
      value: 21.002000000000002
    - type: map_at_5
      value: 22.256
    - type: mrr_at_1
      value: 21.02
    - type: mrr_at_10
      value: 27.898
    - type: mrr_at_100
      value: 29.018
    - type: mrr_at_1000
      value: 29.099999999999998
    - type: mrr_at_3
      value: 25.456
    - type: mrr_at_5
      value: 26.625
    - type: ndcg_at_1
      value: 21.02
    - type: ndcg_at_10
      value: 28.277
    - type: ndcg_at_100
      value: 34.54
    - type: ndcg_at_1000
      value: 37.719
    - type: ndcg_at_3
      value: 23.707
    - type: ndcg_at_5
      value: 25.482
    - type: precision_at_1
      value: 21.02
    - type: precision_at_10
      value: 5.361
    - type: precision_at_100
      value: 0.9809999999999999
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 11.401
    - type: precision_at_5
      value: 8.209
    - type: recall_at_1
      value: 16.928
    - type: recall_at_10
      value: 38.601
    - type: recall_at_100
      value: 65.759
    - type: recall_at_1000
      value: 88.543
    - type: recall_at_3
      value: 25.556
    - type: recall_at_5
      value: 30.447000000000003
    - type: map_at_1
      value: 28.549000000000003
    - type: map_at_10
      value: 38.426
    - type: map_at_100
      value: 39.845000000000006
    - type: map_at_1000
      value: 39.956
    - type: map_at_3
      value: 35.372
    - type: map_at_5
      value: 37.204
    - type: mrr_at_1
      value: 35.034
    - type: mrr_at_10
      value: 44.041000000000004
    - type: mrr_at_100
      value: 44.95
    - type: mrr_at_1000
      value: 44.997
    - type: mrr_at_3
      value: 41.498000000000005
    - type: mrr_at_5
      value: 43.077
    - type: ndcg_at_1
      value: 35.034
    - type: ndcg_at_10
      value: 44.218
    - type: ndcg_at_100
      value: 49.958000000000006
    - type: ndcg_at_1000
      value: 52.019000000000005
    - type: ndcg_at_3
      value: 39.34
    - type: ndcg_at_5
      value: 41.892
    - type: precision_at_1
      value: 35.034
    - type: precision_at_10
      value: 7.911
    - type: precision_at_100
      value: 1.26
    - type: precision_at_1000
      value: 0.16
    - type: precision_at_3
      value: 18.511
    - type: precision_at_5
      value: 13.205
    - type: recall_at_1
      value: 28.549000000000003
    - type: recall_at_10
      value: 56.035999999999994
    - type: recall_at_100
      value: 79.701
    - type: recall_at_1000
      value: 93.149
    - type: recall_at_3
      value: 42.275
    - type: recall_at_5
      value: 49.097
    - type: map_at_1
      value: 29.391000000000002
    - type: map_at_10
      value: 39.48
    - type: map_at_100
      value: 40.727000000000004
    - type: map_at_1000
      value: 40.835
    - type: map_at_3
      value: 36.234
    - type: map_at_5
      value: 37.877
    - type: mrr_at_1
      value: 35.959
    - type: mrr_at_10
      value: 44.726
    - type: mrr_at_100
      value: 45.531
    - type: mrr_at_1000
      value: 45.582
    - type: mrr_at_3
      value: 42.047000000000004
    - type: mrr_at_5
      value: 43.611
    - type: ndcg_at_1
      value: 35.959
    - type: ndcg_at_10
      value: 45.303
    - type: ndcg_at_100
      value: 50.683
    - type: ndcg_at_1000
      value: 52.818
    - type: ndcg_at_3
      value: 39.987
    - type: ndcg_at_5
      value: 42.243
    - type: precision_at_1
      value: 35.959
    - type: precision_at_10
      value: 8.241999999999999
    - type: precision_at_100
      value: 1.274
    - type: precision_at_1000
      value: 0.163
    - type: precision_at_3
      value: 18.836
    - type: precision_at_5
      value: 13.196
    - type: recall_at_1
      value: 29.391000000000002
    - type: recall_at_10
      value: 57.364000000000004
    - type: recall_at_100
      value: 80.683
    - type: recall_at_1000
      value: 94.918
    - type: recall_at_3
      value: 42.263
    - type: recall_at_5
      value: 48.634
    - type: map_at_1
      value: 26.791749999999997
    - type: map_at_10
      value: 35.75541666666667
    - type: map_at_100
      value: 37.00791666666667
    - type: map_at_1000
      value: 37.12408333333333
    - type: map_at_3
      value: 33.02966666666667
    - type: map_at_5
      value: 34.56866666666667
    - type: mrr_at_1
      value: 31.744333333333337
    - type: mrr_at_10
      value: 39.9925
    - type: mrr_at_100
      value: 40.86458333333333
    - type: mrr_at_1000
      value: 40.92175000000001
    - type: mrr_at_3
      value: 37.68183333333334
    - type: mrr_at_5
      value: 39.028499999999994
    - type: ndcg_at_1
      value: 31.744333333333337
    - type: ndcg_at_10
      value: 40.95008333333334
    - type: ndcg_at_100
      value: 46.25966666666667
    - type: ndcg_at_1000
      value: 48.535333333333334
    - type: ndcg_at_3
      value: 36.43333333333333
    - type: ndcg_at_5
      value: 38.602333333333334
    - type: precision_at_1
      value: 31.744333333333337
    - type: precision_at_10
      value: 7.135166666666666
    - type: precision_at_100
      value: 1.1535833333333334
    - type: precision_at_1000
      value: 0.15391666666666665
    - type: precision_at_3
      value: 16.713
    - type: precision_at_5
      value: 11.828416666666666
    - type: recall_at_1
      value: 26.791749999999997
    - type: recall_at_10
      value: 51.98625
    - type: recall_at_100
      value: 75.30358333333334
    - type: recall_at_1000
      value: 91.05433333333333
    - type: recall_at_3
      value: 39.39583333333333
    - type: recall_at_5
      value: 45.05925
    - type: map_at_1
      value: 22.219
    - type: map_at_10
      value: 29.162
    - type: map_at_100
      value: 30.049999999999997
    - type: map_at_1000
      value: 30.144
    - type: map_at_3
      value: 27.204
    - type: map_at_5
      value: 28.351
    - type: mrr_at_1
      value: 25.153
    - type: mrr_at_10
      value: 31.814999999999998
    - type: mrr_at_100
      value: 32.573
    - type: mrr_at_1000
      value: 32.645
    - type: mrr_at_3
      value: 29.934
    - type: mrr_at_5
      value: 30.946
    - type: ndcg_at_1
      value: 25.153
    - type: ndcg_at_10
      value: 33.099000000000004
    - type: ndcg_at_100
      value: 37.768
    - type: ndcg_at_1000
      value: 40.331
    - type: ndcg_at_3
      value: 29.473
    - type: ndcg_at_5
      value: 31.206
    - type: precision_at_1
      value: 25.153
    - type: precision_at_10
      value: 5.183999999999999
    - type: precision_at_100
      value: 0.8170000000000001
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 12.831999999999999
    - type: precision_at_5
      value: 8.895999999999999
    - type: recall_at_1
      value: 22.219
    - type: recall_at_10
      value: 42.637
    - type: recall_at_100
      value: 64.704
    - type: recall_at_1000
      value: 83.963
    - type: recall_at_3
      value: 32.444
    - type: recall_at_5
      value: 36.802
    - type: map_at_1
      value: 17.427999999999997
    - type: map_at_10
      value: 24.029
    - type: map_at_100
      value: 25.119999999999997
    - type: map_at_1000
      value: 25.257
    - type: map_at_3
      value: 22.016
    - type: map_at_5
      value: 23.143
    - type: mrr_at_1
      value: 21.129
    - type: mrr_at_10
      value: 27.750000000000004
    - type: mrr_at_100
      value: 28.666999999999998
    - type: mrr_at_1000
      value: 28.754999999999995
    - type: mrr_at_3
      value: 25.849
    - type: mrr_at_5
      value: 26.939999999999998
    - type: ndcg_at_1
      value: 21.129
    - type: ndcg_at_10
      value: 28.203
    - type: ndcg_at_100
      value: 33.44
    - type: ndcg_at_1000
      value: 36.61
    - type: ndcg_at_3
      value: 24.648999999999997
    - type: ndcg_at_5
      value: 26.316
    - type: precision_at_1
      value: 21.129
    - type: precision_at_10
      value: 5.055
    - type: precision_at_100
      value: 0.909
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 11.666
    - type: precision_at_5
      value: 8.3
    - type: recall_at_1
      value: 17.427999999999997
    - type: recall_at_10
      value: 36.923
    - type: recall_at_100
      value: 60.606
    - type: recall_at_1000
      value: 83.19
    - type: recall_at_3
      value: 26.845000000000002
    - type: recall_at_5
      value: 31.247000000000003
    - type: map_at_1
      value: 26.457000000000004
    - type: map_at_10
      value: 35.228
    - type: map_at_100
      value: 36.475
    - type: map_at_1000
      value: 36.585
    - type: map_at_3
      value: 32.444
    - type: map_at_5
      value: 34.046
    - type: mrr_at_1
      value: 30.784
    - type: mrr_at_10
      value: 39.133
    - type: mrr_at_100
      value: 40.11
    - type: mrr_at_1000
      value: 40.169
    - type: mrr_at_3
      value: 36.692
    - type: mrr_at_5
      value: 38.17
    - type: ndcg_at_1
      value: 30.784
    - type: ndcg_at_10
      value: 40.358
    - type: ndcg_at_100
      value: 46.119
    - type: ndcg_at_1000
      value: 48.428
    - type: ndcg_at_3
      value: 35.504000000000005
    - type: ndcg_at_5
      value: 37.864
    - type: precision_at_1
      value: 30.784
    - type: precision_at_10
      value: 6.800000000000001
    - type: precision_at_100
      value: 1.083
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 15.920000000000002
    - type: precision_at_5
      value: 11.437
    - type: recall_at_1
      value: 26.457000000000004
    - type: recall_at_10
      value: 51.845
    - type: recall_at_100
      value: 77.046
    - type: recall_at_1000
      value: 92.892
    - type: recall_at_3
      value: 38.89
    - type: recall_at_5
      value: 44.688
    - type: map_at_1
      value: 29.378999999999998
    - type: map_at_10
      value: 37.373
    - type: map_at_100
      value: 39.107
    - type: map_at_1000
      value: 39.317
    - type: map_at_3
      value: 34.563
    - type: map_at_5
      value: 36.173
    - type: mrr_at_1
      value: 35.178
    - type: mrr_at_10
      value: 42.44
    - type: mrr_at_100
      value: 43.434
    - type: mrr_at_1000
      value: 43.482
    - type: mrr_at_3
      value: 39.987
    - type: mrr_at_5
      value: 41.370000000000005
    - type: ndcg_at_1
      value: 35.178
    - type: ndcg_at_10
      value: 42.82
    - type: ndcg_at_100
      value: 48.935
    - type: ndcg_at_1000
      value: 51.28
    - type: ndcg_at_3
      value: 38.562999999999995
    - type: ndcg_at_5
      value: 40.687
    - type: precision_at_1
      value: 35.178
    - type: precision_at_10
      value: 7.945
    - type: precision_at_100
      value: 1.524
    - type: precision_at_1000
      value: 0.242
    - type: precision_at_3
      value: 17.721
    - type: precision_at_5
      value: 12.925
    - type: recall_at_1
      value: 29.378999999999998
    - type: recall_at_10
      value: 52.141999999999996
    - type: recall_at_100
      value: 79.49000000000001
    - type: recall_at_1000
      value: 93.782
    - type: recall_at_3
      value: 39.579
    - type: recall_at_5
      value: 45.462
    - type: map_at_1
      value: 19.814999999999998
    - type: map_at_10
      value: 27.383999999999997
    - type: map_at_100
      value: 28.483999999999998
    - type: map_at_1000
      value: 28.585
    - type: map_at_3
      value: 24.807000000000002
    - type: map_at_5
      value: 26.485999999999997
    - type: mrr_at_1
      value: 21.996
    - type: mrr_at_10
      value: 29.584
    - type: mrr_at_100
      value: 30.611
    - type: mrr_at_1000
      value: 30.684
    - type: mrr_at_3
      value: 27.11
    - type: mrr_at_5
      value: 28.746
    - type: ndcg_at_1
      value: 21.996
    - type: ndcg_at_10
      value: 32.024
    - type: ndcg_at_100
      value: 37.528
    - type: ndcg_at_1000
      value: 40.150999999999996
    - type: ndcg_at_3
      value: 27.016000000000002
    - type: ndcg_at_5
      value: 29.927999999999997
    - type: precision_at_1
      value: 21.996
    - type: precision_at_10
      value: 5.102
    - type: precision_at_100
      value: 0.856
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 11.583
    - type: precision_at_5
      value: 8.577
    - type: recall_at_1
      value: 19.814999999999998
    - type: recall_at_10
      value: 44.239
    - type: recall_at_100
      value: 69.269
    - type: recall_at_1000
      value: 89.216
    - type: recall_at_3
      value: 31.102999999999998
    - type: recall_at_5
      value: 38.078
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
      value: 11.349
    - type: map_at_10
      value: 19.436
    - type: map_at_100
      value: 21.282999999999998
    - type: map_at_1000
      value: 21.479
    - type: map_at_3
      value: 15.841
    - type: map_at_5
      value: 17.558
    - type: mrr_at_1
      value: 25.863000000000003
    - type: mrr_at_10
      value: 37.218
    - type: mrr_at_100
      value: 38.198
    - type: mrr_at_1000
      value: 38.236
    - type: mrr_at_3
      value: 33.409
    - type: mrr_at_5
      value: 35.602000000000004
    - type: ndcg_at_1
      value: 25.863000000000003
    - type: ndcg_at_10
      value: 27.953
    - type: ndcg_at_100
      value: 35.327
    - type: ndcg_at_1000
      value: 38.708999999999996
    - type: ndcg_at_3
      value: 21.985
    - type: ndcg_at_5
      value: 23.957
    - type: precision_at_1
      value: 25.863000000000003
    - type: precision_at_10
      value: 8.99
    - type: precision_at_100
      value: 1.6889999999999998
    - type: precision_at_1000
      value: 0.232
    - type: precision_at_3
      value: 16.308
    - type: precision_at_5
      value: 12.912
    - type: recall_at_1
      value: 11.349
    - type: recall_at_10
      value: 34.581
    - type: recall_at_100
      value: 60.178
    - type: recall_at_1000
      value: 78.88199999999999
    - type: recall_at_3
      value: 20.041999999999998
    - type: recall_at_5
      value: 25.458
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
      value: 7.893
    - type: map_at_10
      value: 15.457
    - type: map_at_100
      value: 20.905
    - type: map_at_1000
      value: 22.116
    - type: map_at_3
      value: 11.593
    - type: map_at_5
      value: 13.134
    - type: mrr_at_1
      value: 57.49999999999999
    - type: mrr_at_10
      value: 65.467
    - type: mrr_at_100
      value: 66.022
    - type: mrr_at_1000
      value: 66.039
    - type: mrr_at_3
      value: 63.458000000000006
    - type: mrr_at_5
      value: 64.546
    - type: ndcg_at_1
      value: 45.875
    - type: ndcg_at_10
      value: 33.344
    - type: ndcg_at_100
      value: 36.849
    - type: ndcg_at_1000
      value: 44.03
    - type: ndcg_at_3
      value: 37.504
    - type: ndcg_at_5
      value: 34.892
    - type: precision_at_1
      value: 57.49999999999999
    - type: precision_at_10
      value: 25.95
    - type: precision_at_100
      value: 7.89
    - type: precision_at_1000
      value: 1.669
    - type: precision_at_3
      value: 40.333000000000006
    - type: precision_at_5
      value: 33.050000000000004
    - type: recall_at_1
      value: 7.893
    - type: recall_at_10
      value: 20.724999999999998
    - type: recall_at_100
      value: 42.516
    - type: recall_at_1000
      value: 65.822
    - type: recall_at_3
      value: 12.615000000000002
    - type: recall_at_5
      value: 15.482000000000001
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
      value: 51.760000000000005
    - type: f1
      value: 45.51690565701713
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
      value: 53.882
    - type: map_at_10
      value: 65.902
    - type: map_at_100
      value: 66.33
    - type: map_at_1000
      value: 66.348
    - type: map_at_3
      value: 63.75999999999999
    - type: map_at_5
      value: 65.181
    - type: mrr_at_1
      value: 58.041
    - type: mrr_at_10
      value: 70.133
    - type: mrr_at_100
      value: 70.463
    - type: mrr_at_1000
      value: 70.47
    - type: mrr_at_3
      value: 68.164
    - type: mrr_at_5
      value: 69.465
    - type: ndcg_at_1
      value: 58.041
    - type: ndcg_at_10
      value: 71.84700000000001
    - type: ndcg_at_100
      value: 73.699
    - type: ndcg_at_1000
      value: 74.06700000000001
    - type: ndcg_at_3
      value: 67.855
    - type: ndcg_at_5
      value: 70.203
    - type: precision_at_1
      value: 58.041
    - type: precision_at_10
      value: 9.427000000000001
    - type: precision_at_100
      value: 1.049
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 27.278000000000002
    - type: precision_at_5
      value: 17.693
    - type: recall_at_1
      value: 53.882
    - type: recall_at_10
      value: 85.99
    - type: recall_at_100
      value: 94.09100000000001
    - type: recall_at_1000
      value: 96.612
    - type: recall_at_3
      value: 75.25
    - type: recall_at_5
      value: 80.997
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
      value: 19.165
    - type: map_at_10
      value: 31.845000000000002
    - type: map_at_100
      value: 33.678999999999995
    - type: map_at_1000
      value: 33.878
    - type: map_at_3
      value: 27.881
    - type: map_at_5
      value: 30.049999999999997
    - type: mrr_at_1
      value: 38.272
    - type: mrr_at_10
      value: 47.04
    - type: mrr_at_100
      value: 47.923
    - type: mrr_at_1000
      value: 47.973
    - type: mrr_at_3
      value: 44.985
    - type: mrr_at_5
      value: 46.150000000000006
    - type: ndcg_at_1
      value: 38.272
    - type: ndcg_at_10
      value: 39.177
    - type: ndcg_at_100
      value: 45.995000000000005
    - type: ndcg_at_1000
      value: 49.312
    - type: ndcg_at_3
      value: 36.135
    - type: ndcg_at_5
      value: 36.936
    - type: precision_at_1
      value: 38.272
    - type: precision_at_10
      value: 10.926
    - type: precision_at_100
      value: 1.809
    - type: precision_at_1000
      value: 0.23700000000000002
    - type: precision_at_3
      value: 24.331
    - type: precision_at_5
      value: 17.747
    - type: recall_at_1
      value: 19.165
    - type: recall_at_10
      value: 45.103
    - type: recall_at_100
      value: 70.295
    - type: recall_at_1000
      value: 90.592
    - type: recall_at_3
      value: 32.832
    - type: recall_at_5
      value: 37.905
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
      value: 32.397
    - type: map_at_10
      value: 44.83
    - type: map_at_100
      value: 45.716
    - type: map_at_1000
      value: 45.797
    - type: map_at_3
      value: 41.955999999999996
    - type: map_at_5
      value: 43.736999999999995
    - type: mrr_at_1
      value: 64.794
    - type: mrr_at_10
      value: 71.866
    - type: mrr_at_100
      value: 72.22
    - type: mrr_at_1000
      value: 72.238
    - type: mrr_at_3
      value: 70.416
    - type: mrr_at_5
      value: 71.304
    - type: ndcg_at_1
      value: 64.794
    - type: ndcg_at_10
      value: 54.186
    - type: ndcg_at_100
      value: 57.623000000000005
    - type: ndcg_at_1000
      value: 59.302
    - type: ndcg_at_3
      value: 49.703
    - type: ndcg_at_5
      value: 52.154999999999994
    - type: precision_at_1
      value: 64.794
    - type: precision_at_10
      value: 11.219
    - type: precision_at_100
      value: 1.394
    - type: precision_at_1000
      value: 0.16199999999999998
    - type: precision_at_3
      value: 30.767
    - type: precision_at_5
      value: 20.397000000000002
    - type: recall_at_1
      value: 32.397
    - type: recall_at_10
      value: 56.096999999999994
    - type: recall_at_100
      value: 69.696
    - type: recall_at_1000
      value: 80.88499999999999
    - type: recall_at_3
      value: 46.150999999999996
    - type: recall_at_5
      value: 50.993
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
      value: 81.1744
    - type: ap
      value: 75.44973697032414
    - type: f1
      value: 81.09901117955782
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
      value: 19.519000000000002
    - type: map_at_10
      value: 31.025000000000002
    - type: map_at_100
      value: 32.275999999999996
    - type: map_at_1000
      value: 32.329
    - type: map_at_3
      value: 27.132
    - type: map_at_5
      value: 29.415999999999997
    - type: mrr_at_1
      value: 20.115
    - type: mrr_at_10
      value: 31.569000000000003
    - type: mrr_at_100
      value: 32.768
    - type: mrr_at_1000
      value: 32.816
    - type: mrr_at_3
      value: 27.748
    - type: mrr_at_5
      value: 29.956
    - type: ndcg_at_1
      value: 20.115
    - type: ndcg_at_10
      value: 37.756
    - type: ndcg_at_100
      value: 43.858000000000004
    - type: ndcg_at_1000
      value: 45.199
    - type: ndcg_at_3
      value: 29.818
    - type: ndcg_at_5
      value: 33.875
    - type: precision_at_1
      value: 20.115
    - type: precision_at_10
      value: 6.122
    - type: precision_at_100
      value: 0.919
    - type: precision_at_1000
      value: 0.10300000000000001
    - type: precision_at_3
      value: 12.794
    - type: precision_at_5
      value: 9.731
    - type: recall_at_1
      value: 19.519000000000002
    - type: recall_at_10
      value: 58.62500000000001
    - type: recall_at_100
      value: 86.99
    - type: recall_at_1000
      value: 97.268
    - type: recall_at_3
      value: 37.002
    - type: recall_at_5
      value: 46.778
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
      value: 93.71865025079799
    - type: f1
      value: 93.38906173610519
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
      value: 70.2576379388965
    - type: f1
      value: 49.20405830249464
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
      value: 67.48486886348351
    - type: f1
      value: 64.92199176095157
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
      value: 72.59246805648958
    - type: f1
      value: 72.1222026389164
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
      value: 30.887642595096825
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
      value: 28.3764418784054
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
      value: 31.81544126336991
    - type: mrr
      value: 32.82666576268031
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
      value: 5.185
    - type: map_at_10
      value: 11.158
    - type: map_at_100
      value: 14.041
    - type: map_at_1000
      value: 15.360999999999999
    - type: map_at_3
      value: 8.417
    - type: map_at_5
      value: 9.378
    - type: mrr_at_1
      value: 44.582
    - type: mrr_at_10
      value: 53.083999999999996
    - type: mrr_at_100
      value: 53.787
    - type: mrr_at_1000
      value: 53.824000000000005
    - type: mrr_at_3
      value: 51.187000000000005
    - type: mrr_at_5
      value: 52.379
    - type: ndcg_at_1
      value: 42.57
    - type: ndcg_at_10
      value: 31.593
    - type: ndcg_at_100
      value: 29.093999999999998
    - type: ndcg_at_1000
      value: 37.909
    - type: ndcg_at_3
      value: 37.083
    - type: ndcg_at_5
      value: 34.397
    - type: precision_at_1
      value: 43.963
    - type: precision_at_10
      value: 23.498
    - type: precision_at_100
      value: 7.6160000000000005
    - type: precision_at_1000
      value: 2.032
    - type: precision_at_3
      value: 34.572
    - type: precision_at_5
      value: 29.412
    - type: recall_at_1
      value: 5.185
    - type: recall_at_10
      value: 15.234
    - type: recall_at_100
      value: 29.49
    - type: recall_at_1000
      value: 62.273999999999994
    - type: recall_at_3
      value: 9.55
    - type: recall_at_5
      value: 11.103
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
      value: 23.803
    - type: map_at_10
      value: 38.183
    - type: map_at_100
      value: 39.421
    - type: map_at_1000
      value: 39.464
    - type: map_at_3
      value: 33.835
    - type: map_at_5
      value: 36.327
    - type: mrr_at_1
      value: 26.68
    - type: mrr_at_10
      value: 40.439
    - type: mrr_at_100
      value: 41.415
    - type: mrr_at_1000
      value: 41.443999999999996
    - type: mrr_at_3
      value: 36.612
    - type: mrr_at_5
      value: 38.877
    - type: ndcg_at_1
      value: 26.68
    - type: ndcg_at_10
      value: 45.882
    - type: ndcg_at_100
      value: 51.227999999999994
    - type: ndcg_at_1000
      value: 52.207
    - type: ndcg_at_3
      value: 37.511
    - type: ndcg_at_5
      value: 41.749
    - type: precision_at_1
      value: 26.68
    - type: precision_at_10
      value: 7.9750000000000005
    - type: precision_at_100
      value: 1.0959999999999999
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 17.449
    - type: precision_at_5
      value: 12.897
    - type: recall_at_1
      value: 23.803
    - type: recall_at_10
      value: 67.152
    - type: recall_at_100
      value: 90.522
    - type: recall_at_1000
      value: 97.743
    - type: recall_at_3
      value: 45.338
    - type: recall_at_5
      value: 55.106
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
      value: 70.473
    - type: map_at_10
      value: 84.452
    - type: map_at_100
      value: 85.101
    - type: map_at_1000
      value: 85.115
    - type: map_at_3
      value: 81.435
    - type: map_at_5
      value: 83.338
    - type: mrr_at_1
      value: 81.19
    - type: mrr_at_10
      value: 87.324
    - type: mrr_at_100
      value: 87.434
    - type: mrr_at_1000
      value: 87.435
    - type: mrr_at_3
      value: 86.31
    - type: mrr_at_5
      value: 87.002
    - type: ndcg_at_1
      value: 81.21000000000001
    - type: ndcg_at_10
      value: 88.19
    - type: ndcg_at_100
      value: 89.44
    - type: ndcg_at_1000
      value: 89.526
    - type: ndcg_at_3
      value: 85.237
    - type: ndcg_at_5
      value: 86.892
    - type: precision_at_1
      value: 81.21000000000001
    - type: precision_at_10
      value: 13.417000000000002
    - type: precision_at_100
      value: 1.537
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.31
    - type: precision_at_5
      value: 24.59
    - type: recall_at_1
      value: 70.473
    - type: recall_at_10
      value: 95.367
    - type: recall_at_100
      value: 99.616
    - type: recall_at_1000
      value: 99.996
    - type: recall_at_3
      value: 86.936
    - type: recall_at_5
      value: 91.557
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
      value: 59.25776525253911
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
      value: 63.22135271663078
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
      value: 4.003
    - type: map_at_10
      value: 10.062999999999999
    - type: map_at_100
      value: 11.854000000000001
    - type: map_at_1000
      value: 12.145999999999999
    - type: map_at_3
      value: 7.242
    - type: map_at_5
      value: 8.652999999999999
    - type: mrr_at_1
      value: 19.7
    - type: mrr_at_10
      value: 29.721999999999998
    - type: mrr_at_100
      value: 30.867
    - type: mrr_at_1000
      value: 30.944
    - type: mrr_at_3
      value: 26.683
    - type: mrr_at_5
      value: 28.498
    - type: ndcg_at_1
      value: 19.7
    - type: ndcg_at_10
      value: 17.095
    - type: ndcg_at_100
      value: 24.375
    - type: ndcg_at_1000
      value: 29.831000000000003
    - type: ndcg_at_3
      value: 16.305
    - type: ndcg_at_5
      value: 14.291
    - type: precision_at_1
      value: 19.7
    - type: precision_at_10
      value: 8.799999999999999
    - type: precision_at_100
      value: 1.9349999999999998
    - type: precision_at_1000
      value: 0.32399999999999995
    - type: precision_at_3
      value: 15.2
    - type: precision_at_5
      value: 12.540000000000001
    - type: recall_at_1
      value: 4.003
    - type: recall_at_10
      value: 17.877000000000002
    - type: recall_at_100
      value: 39.217
    - type: recall_at_1000
      value: 65.862
    - type: recall_at_3
      value: 9.242
    - type: recall_at_5
      value: 12.715000000000002
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
      value: 80.25888668589654
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
      value: 77.02037527837669
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
      value: 86.58432681008449
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
      value: 81.31697756099051
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
      value: 88.18867599667057
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
      value: 84.87853941747623
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
      value: 89.46479925383916
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
      value: 66.45272113649146
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
      value: 86.43357313527851
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
      value: 78.82761687254882
    - type: mrr
      value: 93.46223674655047
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
      value: 44.583
    - type: map_at_10
      value: 52.978
    - type: map_at_100
      value: 53.803
    - type: map_at_1000
      value: 53.839999999999996
    - type: map_at_3
      value: 50.03300000000001
    - type: map_at_5
      value: 51.939
    - type: mrr_at_1
      value: 47.0
    - type: mrr_at_10
      value: 54.730000000000004
    - type: mrr_at_100
      value: 55.31399999999999
    - type: mrr_at_1000
      value: 55.346
    - type: mrr_at_3
      value: 52.0
    - type: mrr_at_5
      value: 53.783
    - type: ndcg_at_1
      value: 47.0
    - type: ndcg_at_10
      value: 57.82899999999999
    - type: ndcg_at_100
      value: 61.49400000000001
    - type: ndcg_at_1000
      value: 62.676
    - type: ndcg_at_3
      value: 52.373000000000005
    - type: ndcg_at_5
      value: 55.481
    - type: precision_at_1
      value: 47.0
    - type: precision_at_10
      value: 7.867
    - type: precision_at_100
      value: 0.997
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 20.556
    - type: precision_at_5
      value: 14.066999999999998
    - type: recall_at_1
      value: 44.583
    - type: recall_at_10
      value: 71.172
    - type: recall_at_100
      value: 87.7
    - type: recall_at_1000
      value: 97.333
    - type: recall_at_3
      value: 56.511
    - type: recall_at_5
      value: 64.206
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
      value: 99.66237623762376
    - type: cos_sim_ap
      value: 90.35465126226322
    - type: cos_sim_f1
      value: 82.44575936883628
    - type: cos_sim_precision
      value: 81.32295719844358
    - type: cos_sim_recall
      value: 83.6
    - type: dot_accuracy
      value: 99.66237623762376
    - type: dot_ap
      value: 90.35464287920453
    - type: dot_f1
      value: 82.44575936883628
    - type: dot_precision
      value: 81.32295719844358
    - type: dot_recall
      value: 83.6
    - type: euclidean_accuracy
      value: 99.66237623762376
    - type: euclidean_ap
      value: 90.3546512622632
    - type: euclidean_f1
      value: 82.44575936883628
    - type: euclidean_precision
      value: 81.32295719844358
    - type: euclidean_recall
      value: 83.6
    - type: manhattan_accuracy
      value: 99.65940594059406
    - type: manhattan_ap
      value: 90.29220174849843
    - type: manhattan_f1
      value: 82.4987605354487
    - type: manhattan_precision
      value: 81.80924287118977
    - type: manhattan_recall
      value: 83.2
    - type: max_accuracy
      value: 99.66237623762376
    - type: max_ap
      value: 90.35465126226322
    - type: max_f1
      value: 82.4987605354487
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
      value: 65.0394225901397
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
      value: 35.27954189859326
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
      value: 50.99055979974896
    - type: mrr
      value: 51.82745257193787
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
      value: 30.21655465344237
    - type: cos_sim_spearman
      value: 29.853205339630172
    - type: dot_pearson
      value: 30.216540628083564
    - type: dot_spearman
      value: 29.868978894753027
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
      value: 0.2
    - type: map_at_10
      value: 1.398
    - type: map_at_100
      value: 7.406
    - type: map_at_1000
      value: 18.401
    - type: map_at_3
      value: 0.479
    - type: map_at_5
      value: 0.772
    - type: mrr_at_1
      value: 70.0
    - type: mrr_at_10
      value: 79.25999999999999
    - type: mrr_at_100
      value: 79.25999999999999
    - type: mrr_at_1000
      value: 79.25999999999999
    - type: mrr_at_3
      value: 77.333
    - type: mrr_at_5
      value: 78.133
    - type: ndcg_at_1
      value: 63.0
    - type: ndcg_at_10
      value: 58.548
    - type: ndcg_at_100
      value: 45.216
    - type: ndcg_at_1000
      value: 41.149
    - type: ndcg_at_3
      value: 60.641999999999996
    - type: ndcg_at_5
      value: 61.135
    - type: precision_at_1
      value: 70.0
    - type: precision_at_10
      value: 64.0
    - type: precision_at_100
      value: 46.92
    - type: precision_at_1000
      value: 18.642
    - type: precision_at_3
      value: 64.667
    - type: precision_at_5
      value: 66.4
    - type: recall_at_1
      value: 0.2
    - type: recall_at_10
      value: 1.6729999999999998
    - type: recall_at_100
      value: 10.856
    - type: recall_at_1000
      value: 38.964999999999996
    - type: recall_at_3
      value: 0.504
    - type: recall_at_5
      value: 0.852
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
      value: 1.6629999999999998
    - type: map_at_10
      value: 8.601
    - type: map_at_100
      value: 14.354
    - type: map_at_1000
      value: 15.927
    - type: map_at_3
      value: 4.1930000000000005
    - type: map_at_5
      value: 5.655
    - type: mrr_at_1
      value: 18.367
    - type: mrr_at_10
      value: 34.466
    - type: mrr_at_100
      value: 35.235
    - type: mrr_at_1000
      value: 35.27
    - type: mrr_at_3
      value: 28.571
    - type: mrr_at_5
      value: 31.531
    - type: ndcg_at_1
      value: 14.285999999999998
    - type: ndcg_at_10
      value: 20.374
    - type: ndcg_at_100
      value: 33.532000000000004
    - type: ndcg_at_1000
      value: 45.561
    - type: ndcg_at_3
      value: 18.442
    - type: ndcg_at_5
      value: 18.076
    - type: precision_at_1
      value: 18.367
    - type: precision_at_10
      value: 20.204
    - type: precision_at_100
      value: 7.489999999999999
    - type: precision_at_1000
      value: 1.5630000000000002
    - type: precision_at_3
      value: 21.769
    - type: precision_at_5
      value: 20.408
    - type: recall_at_1
      value: 1.6629999999999998
    - type: recall_at_10
      value: 15.549
    - type: recall_at_100
      value: 47.497
    - type: recall_at_1000
      value: 84.524
    - type: recall_at_3
      value: 5.289
    - type: recall_at_5
      value: 8.035
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
      value: 71.8194
    - type: ap
      value: 14.447702451658554
    - type: f1
      value: 55.13659412856185
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
      value: 63.310696095076416
    - type: f1
      value: 63.360434851097814
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
      value: 51.30677907335145
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
      value: 86.12386004649221
    - type: cos_sim_ap
      value: 73.99096426215495
    - type: cos_sim_f1
      value: 68.18416968442834
    - type: cos_sim_precision
      value: 66.86960933536275
    - type: cos_sim_recall
      value: 69.55145118733509
    - type: dot_accuracy
      value: 86.12386004649221
    - type: dot_ap
      value: 73.99096813038672
    - type: dot_f1
      value: 68.18416968442834
    - type: dot_precision
      value: 66.86960933536275
    - type: dot_recall
      value: 69.55145118733509
    - type: euclidean_accuracy
      value: 86.12386004649221
    - type: euclidean_ap
      value: 73.99095984980165
    - type: euclidean_f1
      value: 68.18416968442834
    - type: euclidean_precision
      value: 66.86960933536275
    - type: euclidean_recall
      value: 69.55145118733509
    - type: manhattan_accuracy
      value: 86.09405734040651
    - type: manhattan_ap
      value: 73.96825745608601
    - type: manhattan_f1
      value: 68.13888179729383
    - type: manhattan_precision
      value: 65.99901088031652
    - type: manhattan_recall
      value: 70.42216358839049
    - type: max_accuracy
      value: 86.12386004649221
    - type: max_ap
      value: 73.99096813038672
    - type: max_f1
      value: 68.18416968442834
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
      value: 88.99367407924865
    - type: cos_sim_ap
      value: 86.19720829843081
    - type: cos_sim_f1
      value: 78.39889075384951
    - type: cos_sim_precision
      value: 74.5110278818144
    - type: cos_sim_recall
      value: 82.71481367416075
    - type: dot_accuracy
      value: 88.99367407924865
    - type: dot_ap
      value: 86.19718471454047
    - type: dot_f1
      value: 78.39889075384951
    - type: dot_precision
      value: 74.5110278818144
    - type: dot_recall
      value: 82.71481367416075
    - type: euclidean_accuracy
      value: 88.99367407924865
    - type: euclidean_ap
      value: 86.1972021422436
    - type: euclidean_f1
      value: 78.39889075384951
    - type: euclidean_precision
      value: 74.5110278818144
    - type: euclidean_recall
      value: 82.71481367416075
    - type: manhattan_accuracy
      value: 88.95680521597392
    - type: manhattan_ap
      value: 86.16659921351506
    - type: manhattan_f1
      value: 78.39125971550081
    - type: manhattan_precision
      value: 74.82502799552073
    - type: manhattan_recall
      value: 82.31444410224823
    - type: max_accuracy
      value: 88.99367407924865
    - type: max_ap
      value: 86.19720829843081
    - type: max_f1
      value: 78.39889075384951
---

# hkunlp/instructor-base
We introduce **Instructor**👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) ***by simply providing the task instruction, without any finetuning***. Instructor👨‍ achieves sota on 70 diverse embedding tasks!
The model is easy to use with **our customized** `sentence-transformer` library. For more details, check out [our paper](https://arxiv.org/abs/2212.09741) and [project page](https://instructor-embedding.github.io/)! 

**************************** **Updates** ****************************

* 01/21: We released a new [checkpoint](https://huggingface.co/hkunlp/instructor-base) trained with hard negatives, which gives better performance.
* 12/21: We released our [paper](https://arxiv.org/abs/2212.09741), [code](https://github.com/HKUNLP/instructor-embedding), [checkpoint](https://huggingface.co/hkunlp/instructor-base) and [project page](https://instructor-embedding.github.io/)! Check them out!

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
model = INSTRUCTOR('hkunlp/instructor-base')
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