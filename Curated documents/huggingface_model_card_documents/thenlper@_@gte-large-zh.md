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
- name: gte-large-zh
  results:
  - task:
      type: STS
    dataset:
      name: MTEB AFQMC
      type: C-MTEB/AFQMC
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 48.94131905219026
    - type: cos_sim_spearman
      value: 54.58261199731436
    - type: euclidean_pearson
      value: 52.73929210805982
    - type: euclidean_spearman
      value: 54.582632097533676
    - type: manhattan_pearson
      value: 52.73123295724949
    - type: manhattan_spearman
      value: 54.572941830465794
  - task:
      type: STS
    dataset:
      name: MTEB ATEC
      type: C-MTEB/ATEC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 47.292931669579005
    - type: cos_sim_spearman
      value: 54.601019783506466
    - type: euclidean_pearson
      value: 54.61393532658173
    - type: euclidean_spearman
      value: 54.60101865708542
    - type: manhattan_pearson
      value: 54.59369555606305
    - type: manhattan_spearman
      value: 54.601098593646036
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (zh)
      type: mteb/amazon_reviews_multi
      config: zh
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 47.233999999999995
    - type: f1
      value: 45.68998446563349
  - task:
      type: STS
    dataset:
      name: MTEB BQ
      type: C-MTEB/BQ
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 62.55033151404683
    - type: cos_sim_spearman
      value: 64.40573802644984
    - type: euclidean_pearson
      value: 62.93453281081951
    - type: euclidean_spearman
      value: 64.40574149035828
    - type: manhattan_pearson
      value: 62.839969210895816
    - type: manhattan_spearman
      value: 64.30837945045283
  - task:
      type: Clustering
    dataset:
      name: MTEB CLSClusteringP2P
      type: C-MTEB/CLSClusteringP2P
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 42.098169316685045
  - task:
      type: Clustering
    dataset:
      name: MTEB CLSClusteringS2S
      type: C-MTEB/CLSClusteringS2S
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 38.90716707051822
  - task:
      type: Reranking
    dataset:
      name: MTEB CMedQAv1
      type: C-MTEB/CMedQAv1-reranking
      config: default
      split: test
      revision: None
    metrics:
    - type: map
      value: 86.09191911031553
    - type: mrr
      value: 88.6747619047619
  - task:
      type: Reranking
    dataset:
      name: MTEB CMedQAv2
      type: C-MTEB/CMedQAv2-reranking
      config: default
      split: test
      revision: None
    metrics:
    - type: map
      value: 86.45781885502122
    - type: mrr
      value: 89.01591269841269
  - task:
      type: Retrieval
    dataset:
      name: MTEB CmedqaRetrieval
      type: C-MTEB/CmedqaRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 24.215
    - type: map_at_10
      value: 36.498000000000005
    - type: map_at_100
      value: 38.409
    - type: map_at_1000
      value: 38.524
    - type: map_at_3
      value: 32.428000000000004
    - type: map_at_5
      value: 34.664
    - type: mrr_at_1
      value: 36.834
    - type: mrr_at_10
      value: 45.196
    - type: mrr_at_100
      value: 46.214
    - type: mrr_at_1000
      value: 46.259
    - type: mrr_at_3
      value: 42.631
    - type: mrr_at_5
      value: 44.044
    - type: ndcg_at_1
      value: 36.834
    - type: ndcg_at_10
      value: 43.146
    - type: ndcg_at_100
      value: 50.632999999999996
    - type: ndcg_at_1000
      value: 52.608999999999995
    - type: ndcg_at_3
      value: 37.851
    - type: ndcg_at_5
      value: 40.005
    - type: precision_at_1
      value: 36.834
    - type: precision_at_10
      value: 9.647
    - type: precision_at_100
      value: 1.574
    - type: precision_at_1000
      value: 0.183
    - type: precision_at_3
      value: 21.48
    - type: precision_at_5
      value: 15.649
    - type: recall_at_1
      value: 24.215
    - type: recall_at_10
      value: 54.079
    - type: recall_at_100
      value: 84.943
    - type: recall_at_1000
      value: 98.098
    - type: recall_at_3
      value: 38.117000000000004
    - type: recall_at_5
      value: 44.775999999999996
  - task:
      type: PairClassification
    dataset:
      name: MTEB Cmnli
      type: C-MTEB/CMNLI
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_accuracy
      value: 82.51352976548407
    - type: cos_sim_ap
      value: 89.49905141462749
    - type: cos_sim_f1
      value: 83.89334489486234
    - type: cos_sim_precision
      value: 78.19761567993534
    - type: cos_sim_recall
      value: 90.48398410100538
    - type: dot_accuracy
      value: 82.51352976548407
    - type: dot_ap
      value: 89.49108293121158
    - type: dot_f1
      value: 83.89334489486234
    - type: dot_precision
      value: 78.19761567993534
    - type: dot_recall
      value: 90.48398410100538
    - type: euclidean_accuracy
      value: 82.51352976548407
    - type: euclidean_ap
      value: 89.49904709975154
    - type: euclidean_f1
      value: 83.89334489486234
    - type: euclidean_precision
      value: 78.19761567993534
    - type: euclidean_recall
      value: 90.48398410100538
    - type: manhattan_accuracy
      value: 82.48947684906794
    - type: manhattan_ap
      value: 89.49231995962901
    - type: manhattan_f1
      value: 83.84681215233205
    - type: manhattan_precision
      value: 77.28258726089528
    - type: manhattan_recall
      value: 91.62964694879588
    - type: max_accuracy
      value: 82.51352976548407
    - type: max_ap
      value: 89.49905141462749
    - type: max_f1
      value: 83.89334489486234
  - task:
      type: Retrieval
    dataset:
      name: MTEB CovidRetrieval
      type: C-MTEB/CovidRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 78.583
    - type: map_at_10
      value: 85.613
    - type: map_at_100
      value: 85.777
    - type: map_at_1000
      value: 85.77900000000001
    - type: map_at_3
      value: 84.58
    - type: map_at_5
      value: 85.22800000000001
    - type: mrr_at_1
      value: 78.925
    - type: mrr_at_10
      value: 85.667
    - type: mrr_at_100
      value: 85.822
    - type: mrr_at_1000
      value: 85.824
    - type: mrr_at_3
      value: 84.651
    - type: mrr_at_5
      value: 85.299
    - type: ndcg_at_1
      value: 78.925
    - type: ndcg_at_10
      value: 88.405
    - type: ndcg_at_100
      value: 89.02799999999999
    - type: ndcg_at_1000
      value: 89.093
    - type: ndcg_at_3
      value: 86.393
    - type: ndcg_at_5
      value: 87.5
    - type: precision_at_1
      value: 78.925
    - type: precision_at_10
      value: 9.789
    - type: precision_at_100
      value: 1.005
    - type: precision_at_1000
      value: 0.101
    - type: precision_at_3
      value: 30.769000000000002
    - type: precision_at_5
      value: 19.031000000000002
    - type: recall_at_1
      value: 78.583
    - type: recall_at_10
      value: 96.891
    - type: recall_at_100
      value: 99.473
    - type: recall_at_1000
      value: 100.0
    - type: recall_at_3
      value: 91.438
    - type: recall_at_5
      value: 94.152
  - task:
      type: Retrieval
    dataset:
      name: MTEB DuRetrieval
      type: C-MTEB/DuRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 25.604
    - type: map_at_10
      value: 77.171
    - type: map_at_100
      value: 80.033
    - type: map_at_1000
      value: 80.099
    - type: map_at_3
      value: 54.364000000000004
    - type: map_at_5
      value: 68.024
    - type: mrr_at_1
      value: 89.85
    - type: mrr_at_10
      value: 93.009
    - type: mrr_at_100
      value: 93.065
    - type: mrr_at_1000
      value: 93.068
    - type: mrr_at_3
      value: 92.72500000000001
    - type: mrr_at_5
      value: 92.915
    - type: ndcg_at_1
      value: 89.85
    - type: ndcg_at_10
      value: 85.038
    - type: ndcg_at_100
      value: 88.247
    - type: ndcg_at_1000
      value: 88.837
    - type: ndcg_at_3
      value: 85.20299999999999
    - type: ndcg_at_5
      value: 83.47
    - type: precision_at_1
      value: 89.85
    - type: precision_at_10
      value: 40.275
    - type: precision_at_100
      value: 4.709
    - type: precision_at_1000
      value: 0.486
    - type: precision_at_3
      value: 76.36699999999999
    - type: precision_at_5
      value: 63.75999999999999
    - type: recall_at_1
      value: 25.604
    - type: recall_at_10
      value: 85.423
    - type: recall_at_100
      value: 95.695
    - type: recall_at_1000
      value: 98.669
    - type: recall_at_3
      value: 56.737
    - type: recall_at_5
      value: 72.646
  - task:
      type: Retrieval
    dataset:
      name: MTEB EcomRetrieval
      type: C-MTEB/EcomRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 51.800000000000004
    - type: map_at_10
      value: 62.17
    - type: map_at_100
      value: 62.649
    - type: map_at_1000
      value: 62.663000000000004
    - type: map_at_3
      value: 59.699999999999996
    - type: map_at_5
      value: 61.23499999999999
    - type: mrr_at_1
      value: 51.800000000000004
    - type: mrr_at_10
      value: 62.17
    - type: mrr_at_100
      value: 62.649
    - type: mrr_at_1000
      value: 62.663000000000004
    - type: mrr_at_3
      value: 59.699999999999996
    - type: mrr_at_5
      value: 61.23499999999999
    - type: ndcg_at_1
      value: 51.800000000000004
    - type: ndcg_at_10
      value: 67.246
    - type: ndcg_at_100
      value: 69.58
    - type: ndcg_at_1000
      value: 69.925
    - type: ndcg_at_3
      value: 62.197
    - type: ndcg_at_5
      value: 64.981
    - type: precision_at_1
      value: 51.800000000000004
    - type: precision_at_10
      value: 8.32
    - type: precision_at_100
      value: 0.941
    - type: precision_at_1000
      value: 0.097
    - type: precision_at_3
      value: 23.133
    - type: precision_at_5
      value: 15.24
    - type: recall_at_1
      value: 51.800000000000004
    - type: recall_at_10
      value: 83.2
    - type: recall_at_100
      value: 94.1
    - type: recall_at_1000
      value: 96.8
    - type: recall_at_3
      value: 69.39999999999999
    - type: recall_at_5
      value: 76.2
  - task:
      type: Classification
    dataset:
      name: MTEB IFlyTek
      type: C-MTEB/IFlyTek-classification
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 49.60369372835706
    - type: f1
      value: 38.24016248875209
  - task:
      type: Classification
    dataset:
      name: MTEB JDReview
      type: C-MTEB/JDReview-classification
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 86.71669793621012
    - type: ap
      value: 55.75807094995178
    - type: f1
      value: 81.59033162805417
  - task:
      type: STS
    dataset:
      name: MTEB LCQMC
      type: C-MTEB/LCQMC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 69.50947272908907
    - type: cos_sim_spearman
      value: 74.40054474949213
    - type: euclidean_pearson
      value: 73.53007373987617
    - type: euclidean_spearman
      value: 74.40054474732082
    - type: manhattan_pearson
      value: 73.51396571849736
    - type: manhattan_spearman
      value: 74.38395696630835
  - task:
      type: Reranking
    dataset:
      name: MTEB MMarcoReranking
      type: C-MTEB/Mmarco-reranking
      config: default
      split: dev
      revision: None
    metrics:
    - type: map
      value: 31.188333827724108
    - type: mrr
      value: 29.84801587301587
  - task:
      type: Retrieval
    dataset:
      name: MTEB MMarcoRetrieval
      type: C-MTEB/MMarcoRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 64.685
    - type: map_at_10
      value: 73.803
    - type: map_at_100
      value: 74.153
    - type: map_at_1000
      value: 74.167
    - type: map_at_3
      value: 71.98
    - type: map_at_5
      value: 73.21600000000001
    - type: mrr_at_1
      value: 66.891
    - type: mrr_at_10
      value: 74.48700000000001
    - type: mrr_at_100
      value: 74.788
    - type: mrr_at_1000
      value: 74.801
    - type: mrr_at_3
      value: 72.918
    - type: mrr_at_5
      value: 73.965
    - type: ndcg_at_1
      value: 66.891
    - type: ndcg_at_10
      value: 77.534
    - type: ndcg_at_100
      value: 79.106
    - type: ndcg_at_1000
      value: 79.494
    - type: ndcg_at_3
      value: 74.13499999999999
    - type: ndcg_at_5
      value: 76.20700000000001
    - type: precision_at_1
      value: 66.891
    - type: precision_at_10
      value: 9.375
    - type: precision_at_100
      value: 1.0170000000000001
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_3
      value: 27.932000000000002
    - type: precision_at_5
      value: 17.86
    - type: recall_at_1
      value: 64.685
    - type: recall_at_10
      value: 88.298
    - type: recall_at_100
      value: 95.426
    - type: recall_at_1000
      value: 98.48700000000001
    - type: recall_at_3
      value: 79.44200000000001
    - type: recall_at_5
      value: 84.358
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (zh-CN)
      type: mteb/amazon_massive_intent
      config: zh-CN
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 73.30531271015468
    - type: f1
      value: 70.88091430578575
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (zh-CN)
      type: mteb/amazon_massive_scenario
      config: zh-CN
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 75.7128446536651
    - type: f1
      value: 75.06125593532262
  - task:
      type: Retrieval
    dataset:
      name: MTEB MedicalRetrieval
      type: C-MTEB/MedicalRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 52.7
    - type: map_at_10
      value: 59.532
    - type: map_at_100
      value: 60.085
    - type: map_at_1000
      value: 60.126000000000005
    - type: map_at_3
      value: 57.767
    - type: map_at_5
      value: 58.952000000000005
    - type: mrr_at_1
      value: 52.900000000000006
    - type: mrr_at_10
      value: 59.648999999999994
    - type: mrr_at_100
      value: 60.20100000000001
    - type: mrr_at_1000
      value: 60.242
    - type: mrr_at_3
      value: 57.882999999999996
    - type: mrr_at_5
      value: 59.068
    - type: ndcg_at_1
      value: 52.7
    - type: ndcg_at_10
      value: 62.883
    - type: ndcg_at_100
      value: 65.714
    - type: ndcg_at_1000
      value: 66.932
    - type: ndcg_at_3
      value: 59.34700000000001
    - type: ndcg_at_5
      value: 61.486
    - type: precision_at_1
      value: 52.7
    - type: precision_at_10
      value: 7.340000000000001
    - type: precision_at_100
      value: 0.8699999999999999
    - type: precision_at_1000
      value: 0.097
    - type: precision_at_3
      value: 21.3
    - type: precision_at_5
      value: 13.819999999999999
    - type: recall_at_1
      value: 52.7
    - type: recall_at_10
      value: 73.4
    - type: recall_at_100
      value: 87.0
    - type: recall_at_1000
      value: 96.8
    - type: recall_at_3
      value: 63.9
    - type: recall_at_5
      value: 69.1
  - task:
      type: Classification
    dataset:
      name: MTEB MultilingualSentiment
      type: C-MTEB/MultilingualSentiment-classification
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 76.47666666666667
    - type: f1
      value: 76.4808576632057
  - task:
      type: PairClassification
    dataset:
      name: MTEB Ocnli
      type: C-MTEB/OCNLI
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_accuracy
      value: 77.58527341635084
    - type: cos_sim_ap
      value: 79.32131557636497
    - type: cos_sim_f1
      value: 80.51948051948052
    - type: cos_sim_precision
      value: 71.7948717948718
    - type: cos_sim_recall
      value: 91.65786694825766
    - type: dot_accuracy
      value: 77.58527341635084
    - type: dot_ap
      value: 79.32131557636497
    - type: dot_f1
      value: 80.51948051948052
    - type: dot_precision
      value: 71.7948717948718
    - type: dot_recall
      value: 91.65786694825766
    - type: euclidean_accuracy
      value: 77.58527341635084
    - type: euclidean_ap
      value: 79.32131557636497
    - type: euclidean_f1
      value: 80.51948051948052
    - type: euclidean_precision
      value: 71.7948717948718
    - type: euclidean_recall
      value: 91.65786694825766
    - type: manhattan_accuracy
      value: 77.15213860314023
    - type: manhattan_ap
      value: 79.26178519246496
    - type: manhattan_f1
      value: 80.22028453418999
    - type: manhattan_precision
      value: 70.94155844155844
    - type: manhattan_recall
      value: 92.29144667370645
    - type: max_accuracy
      value: 77.58527341635084
    - type: max_ap
      value: 79.32131557636497
    - type: max_f1
      value: 80.51948051948052
  - task:
      type: Classification
    dataset:
      name: MTEB OnlineShopping
      type: C-MTEB/OnlineShopping-classification
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 92.68
    - type: ap
      value: 90.78652757815115
    - type: f1
      value: 92.67153098230253
  - task:
      type: STS
    dataset:
      name: MTEB PAWSX
      type: C-MTEB/PAWSX
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 35.301730226895955
    - type: cos_sim_spearman
      value: 38.54612530948101
    - type: euclidean_pearson
      value: 39.02831131230217
    - type: euclidean_spearman
      value: 38.54612530948101
    - type: manhattan_pearson
      value: 39.04765584936325
    - type: manhattan_spearman
      value: 38.54455759013173
  - task:
      type: STS
    dataset:
      name: MTEB QBQTC
      type: C-MTEB/QBQTC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 32.27907454729754
    - type: cos_sim_spearman
      value: 33.35945567162729
    - type: euclidean_pearson
      value: 31.997628193815725
    - type: euclidean_spearman
      value: 33.3592386340529
    - type: manhattan_pearson
      value: 31.97117833750544
    - type: manhattan_spearman
      value: 33.30857326127779
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (zh)
      type: mteb/sts22-crosslingual-sts
      config: zh
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 62.53712784446981
    - type: cos_sim_spearman
      value: 62.975074386224286
    - type: euclidean_pearson
      value: 61.791207731290854
    - type: euclidean_spearman
      value: 62.975073716988064
    - type: manhattan_pearson
      value: 62.63850653150875
    - type: manhattan_spearman
      value: 63.56640346497343
  - task:
      type: STS
    dataset:
      name: MTEB STSB
      type: C-MTEB/STSB
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 79.52067424748047
    - type: cos_sim_spearman
      value: 79.68425102631514
    - type: euclidean_pearson
      value: 79.27553959329275
    - type: euclidean_spearman
      value: 79.68450427089856
    - type: manhattan_pearson
      value: 79.21584650471131
    - type: manhattan_spearman
      value: 79.6419242840243
  - task:
      type: Reranking
    dataset:
      name: MTEB T2Reranking
      type: C-MTEB/T2Reranking
      config: default
      split: dev
      revision: None
    metrics:
    - type: map
      value: 65.8563449629786
    - type: mrr
      value: 75.82550832339254
  - task:
      type: Retrieval
    dataset:
      name: MTEB T2Retrieval
      type: C-MTEB/T2Retrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 27.889999999999997
    - type: map_at_10
      value: 72.878
    - type: map_at_100
      value: 76.737
    - type: map_at_1000
      value: 76.836
    - type: map_at_3
      value: 52.738
    - type: map_at_5
      value: 63.726000000000006
    - type: mrr_at_1
      value: 89.35600000000001
    - type: mrr_at_10
      value: 92.622
    - type: mrr_at_100
      value: 92.692
    - type: mrr_at_1000
      value: 92.694
    - type: mrr_at_3
      value: 92.13799999999999
    - type: mrr_at_5
      value: 92.452
    - type: ndcg_at_1
      value: 89.35600000000001
    - type: ndcg_at_10
      value: 81.932
    - type: ndcg_at_100
      value: 86.351
    - type: ndcg_at_1000
      value: 87.221
    - type: ndcg_at_3
      value: 84.29100000000001
    - type: ndcg_at_5
      value: 82.279
    - type: precision_at_1
      value: 89.35600000000001
    - type: precision_at_10
      value: 39.511
    - type: precision_at_100
      value: 4.901
    - type: precision_at_1000
      value: 0.513
    - type: precision_at_3
      value: 72.62100000000001
    - type: precision_at_5
      value: 59.918000000000006
    - type: recall_at_1
      value: 27.889999999999997
    - type: recall_at_10
      value: 80.636
    - type: recall_at_100
      value: 94.333
    - type: recall_at_1000
      value: 98.39099999999999
    - type: recall_at_3
      value: 54.797
    - type: recall_at_5
      value: 67.824
  - task:
      type: Classification
    dataset:
      name: MTEB TNews
      type: C-MTEB/TNews-classification
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 51.979000000000006
    - type: f1
      value: 50.35658238894168
  - task:
      type: Clustering
    dataset:
      name: MTEB ThuNewsClusteringP2P
      type: C-MTEB/ThuNewsClusteringP2P
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 68.36477832710159
  - task:
      type: Clustering
    dataset:
      name: MTEB ThuNewsClusteringS2S
      type: C-MTEB/ThuNewsClusteringS2S
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 62.92080622759053
  - task:
      type: Retrieval
    dataset:
      name: MTEB VideoRetrieval
      type: C-MTEB/VideoRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 59.3
    - type: map_at_10
      value: 69.299
    - type: map_at_100
      value: 69.669
    - type: map_at_1000
      value: 69.682
    - type: map_at_3
      value: 67.583
    - type: map_at_5
      value: 68.57799999999999
    - type: mrr_at_1
      value: 59.3
    - type: mrr_at_10
      value: 69.299
    - type: mrr_at_100
      value: 69.669
    - type: mrr_at_1000
      value: 69.682
    - type: mrr_at_3
      value: 67.583
    - type: mrr_at_5
      value: 68.57799999999999
    - type: ndcg_at_1
      value: 59.3
    - type: ndcg_at_10
      value: 73.699
    - type: ndcg_at_100
      value: 75.626
    - type: ndcg_at_1000
      value: 75.949
    - type: ndcg_at_3
      value: 70.18900000000001
    - type: ndcg_at_5
      value: 71.992
    - type: precision_at_1
      value: 59.3
    - type: precision_at_10
      value: 8.73
    - type: precision_at_100
      value: 0.9650000000000001
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 25.900000000000002
    - type: precision_at_5
      value: 16.42
    - type: recall_at_1
      value: 59.3
    - type: recall_at_10
      value: 87.3
    - type: recall_at_100
      value: 96.5
    - type: recall_at_1000
      value: 99.0
    - type: recall_at_3
      value: 77.7
    - type: recall_at_5
      value: 82.1
  - task:
      type: Classification
    dataset:
      name: MTEB Waimai
      type: C-MTEB/waimai-classification
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 88.36999999999999
    - type: ap
      value: 73.29590829222836
    - type: f1
      value: 86.74250506247606
---

# gte-large-zh

General Text Embeddings (GTE) model. [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)

The GTE models are trained by Alibaba DAMO Academy. They are mainly based on the BERT framework and currently offer different sizes of models for both Chinese and English Languages. The GTE models are trained on a large-scale corpus of relevance text pairs, covering a wide range of domains and scenarios. This enables the GTE models to be applied to various downstream tasks of text embeddings, including **information retrieval**, **semantic textual similarity**, **text reranking**, etc.

## Model List

| Models | Language | Max Sequence Length | Dimension | Model Size |
|:-----: | :-----: |:-----: |:-----: |:-----: |
|[GTE-large-zh](https://huggingface.co/thenlper/gte-large-zh) | Chinese | 512 | 1024 | 0.67GB |
|[GTE-base-zh](https://huggingface.co/thenlper/gte-base-zh) | Chinese | 512 | 512 | 0.21GB |
|[GTE-small-zh](https://huggingface.co/thenlper/gte-small-zh) | Chinese | 512 | 512 | 0.10GB |
|[GTE-large](https://huggingface.co/thenlper/gte-large) | English | 512 | 1024 | 0.67GB |
|[GTE-base](https://huggingface.co/thenlper/gte-base) | English | 512 | 512 | 0.21GB |
|[GTE-small](https://huggingface.co/thenlper/gte-small) | English | 512 | 384 | 0.10GB |

## Metrics

We compared the performance of the GTE models with other popular text embedding models on the MTEB (CMTEB for Chinese language) benchmark. For more detailed comparison results, please refer to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

- Evaluation results on CMTEB 

| Model               | Model Size (GB) | Embedding Dimensions | Sequence Length | Average (35 datasets) | Classification  (9 datasets)         | Clustering (4 datasets)        | Pair Classification        (2 datasets) | Reranking (4 datasets)         | Retrieval  (8 datasets)      | STS         (8 datasets) |
| ------------------- | -------------- | -------------------- | ---------------- | --------------------- | ------------------------------------ | ------------------------------ | --------------------------------------- | ------------------------------ | ---------------------------- | ------------------------ |
| **gte-large-zh** | 0.65           | 1024                  | 512              | **66.72**                 | 71.34                                | 53.07                        | 81.14                                   | 67.42                          | 72.49                        | 57.82                  |
| gte-base-zh         | 0.20         | 768                  | 512              | 65.92                 | 71.26                               | 53.86                          | 80.44                                  | 67.00                          | 71.71                        | 55.96                    |
| stella-large-zh-v2  | 0.65           | 1024                 | 1024             | 65.13                 | 69.05                                | 49.16                          | 82.68                                   | 66.41                          | 70.14                        | 58.66                    |
| stella-large-zh     | 0.65           | 1024                 | 1024             | 64.54                 | 67.62                                | 48.65                          | 78.72                                   | 65.98                          | 71.02                        | 58.3                     |
| bge-large-zh-v1.5   | 1.3            | 1024                 | 512              | 64.53                 | 69.13                                | 48.99                          | 81.6                                    | 65.84                          | 70.46                        | 56.25                    |
| stella-base-zh-v2   | 0.21           | 768                  | 1024             | 64.36                 | 68.29                                | 49.4                           | 79.96                                   | 66.1                           | 70.08                        | 56.92                    |
| stella-base-zh      | 0.21           | 768                  | 1024             | 64.16                 | 67.77                                | 48.7                           | 76.09                                   | 66.95                          | 71.07                        | 56.54                    |
| piccolo-large-zh    | 0.65           | 1024                 | 512              | 64.11                 | 67.03                                | 47.04                          | 78.38                                   | 65.98                          | 70.93                        | 58.02                    |
| piccolo-base-zh     | 0.2            | 768                  | 512              | 63.66                 | 66.98                                | 47.12                          | 76.61                                   | 66.68                          | 71.2                         | 55.9                     |
| gte-small-zh         | 0.1           | 512                  | 512              | 60.04                 | 64.35                                | 48.95                          | 69.99                                   | 66.21                          | 65.50                        | 49.72                    |
| bge-small-zh-v1.5     | 0.1           | 512                  | 512              | 57.82                 | 63.96                                | 44.18                          | 70.4                                   | 60.92                          | 61.77                        | 49.1                    |
| m3e-base | 0.41 | 768 | 512 | 57.79 | 67.52 | 47.68 | 63.99 | 59.54| 56.91 | 50.47 | 
|text-embedding-ada-002(openai) | - | 1536| 8192 | 53.02 | 64.31 | 45.68 | 69.56 | 54.28 | 52.0 | 43.35 |


## Usage

Code example

```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

input_texts = [
    "中国的首都是哪里",
    "你喜欢去哪里旅游",
    "北京",
    "今天中午吃什么"
]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large-zh")
model = AutoModel.from_pretrained("thenlper/gte-large-zh")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]
 
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

model = SentenceTransformer('thenlper/gte-large-zh')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
```

### Limitation

This model exclusively caters to Chinese texts, and any lengthy texts will be truncated to a maximum of 512 tokens.

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
