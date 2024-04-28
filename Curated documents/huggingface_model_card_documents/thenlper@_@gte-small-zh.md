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
- name: gte-small-zh
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
      value: 35.80906032378281
    - type: cos_sim_spearman
      value: 36.688967176174415
    - type: euclidean_pearson
      value: 35.70701955438158
    - type: euclidean_spearman
      value: 36.6889470691436
    - type: manhattan_pearson
      value: 35.832741768286944
    - type: manhattan_spearman
      value: 36.831888591957195
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
      value: 44.667266488330384
    - type: cos_sim_spearman
      value: 45.77390794946174
    - type: euclidean_pearson
      value: 48.14272832901943
    - type: euclidean_spearman
      value: 45.77390569666109
    - type: manhattan_pearson
      value: 48.187667158563094
    - type: manhattan_spearman
      value: 45.80979161966117
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
      value: 38.690000000000005
    - type: f1
      value: 36.868257131984016
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
      value: 49.03674224607541
    - type: cos_sim_spearman
      value: 49.63568854885055
    - type: euclidean_pearson
      value: 49.47441886441355
    - type: euclidean_spearman
      value: 49.63567815431205
    - type: manhattan_pearson
      value: 49.76480072909559
    - type: manhattan_spearman
      value: 49.977789367288224
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
      value: 39.538126779019755
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
      value: 37.333105487031766
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
      value: 86.08142426347963
    - type: mrr
      value: 88.04269841269841
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
      value: 87.25694119382474
    - type: mrr
      value: 89.36853174603175
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
      value: 23.913999999999998
    - type: map_at_10
      value: 35.913000000000004
    - type: map_at_100
      value: 37.836
    - type: map_at_1000
      value: 37.952000000000005
    - type: map_at_3
      value: 31.845000000000002
    - type: map_at_5
      value: 34.0
    - type: mrr_at_1
      value: 36.884
    - type: mrr_at_10
      value: 44.872
    - type: mrr_at_100
      value: 45.899
    - type: mrr_at_1000
      value: 45.945
    - type: mrr_at_3
      value: 42.331
    - type: mrr_at_5
      value: 43.674
    - type: ndcg_at_1
      value: 36.884
    - type: ndcg_at_10
      value: 42.459
    - type: ndcg_at_100
      value: 50.046
    - type: ndcg_at_1000
      value: 52.092000000000006
    - type: ndcg_at_3
      value: 37.225
    - type: ndcg_at_5
      value: 39.2
    - type: precision_at_1
      value: 36.884
    - type: precision_at_10
      value: 9.562
    - type: precision_at_100
      value: 1.572
    - type: precision_at_1000
      value: 0.183
    - type: precision_at_3
      value: 21.122
    - type: precision_at_5
      value: 15.274
    - type: recall_at_1
      value: 23.913999999999998
    - type: recall_at_10
      value: 52.891999999999996
    - type: recall_at_100
      value: 84.328
    - type: recall_at_1000
      value: 98.168
    - type: recall_at_3
      value: 37.095
    - type: recall_at_5
      value: 43.396
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
      value: 68.91160553217077
    - type: cos_sim_ap
      value: 76.45769658379533
    - type: cos_sim_f1
      value: 72.07988702844463
    - type: cos_sim_precision
      value: 63.384779137839274
    - type: cos_sim_recall
      value: 83.53986439092822
    - type: dot_accuracy
      value: 68.91160553217077
    - type: dot_ap
      value: 76.47279917239219
    - type: dot_f1
      value: 72.07988702844463
    - type: dot_precision
      value: 63.384779137839274
    - type: dot_recall
      value: 83.53986439092822
    - type: euclidean_accuracy
      value: 68.91160553217077
    - type: euclidean_ap
      value: 76.45768544225383
    - type: euclidean_f1
      value: 72.07988702844463
    - type: euclidean_precision
      value: 63.384779137839274
    - type: euclidean_recall
      value: 83.53986439092822
    - type: manhattan_accuracy
      value: 69.21226698737222
    - type: manhattan_ap
      value: 76.6623683693766
    - type: manhattan_f1
      value: 72.14058164628506
    - type: manhattan_precision
      value: 64.35643564356435
    - type: manhattan_recall
      value: 82.06686930091185
    - type: max_accuracy
      value: 69.21226698737222
    - type: max_ap
      value: 76.6623683693766
    - type: max_f1
      value: 72.14058164628506
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
      value: 48.419000000000004
    - type: map_at_10
      value: 57.367999999999995
    - type: map_at_100
      value: 58.081
    - type: map_at_1000
      value: 58.108000000000004
    - type: map_at_3
      value: 55.251
    - type: map_at_5
      value: 56.53399999999999
    - type: mrr_at_1
      value: 48.472
    - type: mrr_at_10
      value: 57.359
    - type: mrr_at_100
      value: 58.055
    - type: mrr_at_1000
      value: 58.082
    - type: mrr_at_3
      value: 55.303999999999995
    - type: mrr_at_5
      value: 56.542
    - type: ndcg_at_1
      value: 48.472
    - type: ndcg_at_10
      value: 61.651999999999994
    - type: ndcg_at_100
      value: 65.257
    - type: ndcg_at_1000
      value: 65.977
    - type: ndcg_at_3
      value: 57.401
    - type: ndcg_at_5
      value: 59.681
    - type: precision_at_1
      value: 48.472
    - type: precision_at_10
      value: 7.576
    - type: precision_at_100
      value: 0.932
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 21.25
    - type: precision_at_5
      value: 13.888
    - type: recall_at_1
      value: 48.419000000000004
    - type: recall_at_10
      value: 74.97399999999999
    - type: recall_at_100
      value: 92.202
    - type: recall_at_1000
      value: 97.893
    - type: recall_at_3
      value: 63.541000000000004
    - type: recall_at_5
      value: 68.994
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
      value: 22.328
    - type: map_at_10
      value: 69.11
    - type: map_at_100
      value: 72.47
    - type: map_at_1000
      value: 72.54599999999999
    - type: map_at_3
      value: 46.938
    - type: map_at_5
      value: 59.56
    - type: mrr_at_1
      value: 81.35
    - type: mrr_at_10
      value: 87.066
    - type: mrr_at_100
      value: 87.212
    - type: mrr_at_1000
      value: 87.21799999999999
    - type: mrr_at_3
      value: 86.558
    - type: mrr_at_5
      value: 86.931
    - type: ndcg_at_1
      value: 81.35
    - type: ndcg_at_10
      value: 78.568
    - type: ndcg_at_100
      value: 82.86099999999999
    - type: ndcg_at_1000
      value: 83.628
    - type: ndcg_at_3
      value: 76.716
    - type: ndcg_at_5
      value: 75.664
    - type: precision_at_1
      value: 81.35
    - type: precision_at_10
      value: 38.545
    - type: precision_at_100
      value: 4.657
    - type: precision_at_1000
      value: 0.484
    - type: precision_at_3
      value: 69.18299999999999
    - type: precision_at_5
      value: 58.67
    - type: recall_at_1
      value: 22.328
    - type: recall_at_10
      value: 80.658
    - type: recall_at_100
      value: 94.093
    - type: recall_at_1000
      value: 98.137
    - type: recall_at_3
      value: 50.260000000000005
    - type: recall_at_5
      value: 66.045
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
      value: 43.1
    - type: map_at_10
      value: 52.872
    - type: map_at_100
      value: 53.556000000000004
    - type: map_at_1000
      value: 53.583000000000006
    - type: map_at_3
      value: 50.14999999999999
    - type: map_at_5
      value: 51.925
    - type: mrr_at_1
      value: 43.1
    - type: mrr_at_10
      value: 52.872
    - type: mrr_at_100
      value: 53.556000000000004
    - type: mrr_at_1000
      value: 53.583000000000006
    - type: mrr_at_3
      value: 50.14999999999999
    - type: mrr_at_5
      value: 51.925
    - type: ndcg_at_1
      value: 43.1
    - type: ndcg_at_10
      value: 57.907
    - type: ndcg_at_100
      value: 61.517999999999994
    - type: ndcg_at_1000
      value: 62.175000000000004
    - type: ndcg_at_3
      value: 52.425
    - type: ndcg_at_5
      value: 55.631
    - type: precision_at_1
      value: 43.1
    - type: precision_at_10
      value: 7.380000000000001
    - type: precision_at_100
      value: 0.9129999999999999
    - type: precision_at_1000
      value: 0.096
    - type: precision_at_3
      value: 19.667
    - type: precision_at_5
      value: 13.36
    - type: recall_at_1
      value: 43.1
    - type: recall_at_10
      value: 73.8
    - type: recall_at_100
      value: 91.3
    - type: recall_at_1000
      value: 96.39999999999999
    - type: recall_at_3
      value: 59.0
    - type: recall_at_5
      value: 66.8
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
      value: 41.146594844170835
    - type: f1
      value: 28.544218732704845
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
      value: 82.83302063789868
    - type: ap
      value: 48.881798834997056
    - type: f1
      value: 77.28655923994657
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
      value: 66.05467125345538
    - type: cos_sim_spearman
      value: 72.71921060562211
    - type: euclidean_pearson
      value: 71.28539457113986
    - type: euclidean_spearman
      value: 72.71920173126693
    - type: manhattan_pearson
      value: 71.23750818174456
    - type: manhattan_spearman
      value: 72.61025268693467
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
      value: 26.127712982639483
    - type: mrr
      value: 24.87420634920635
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
      value: 62.517
    - type: map_at_10
      value: 71.251
    - type: map_at_100
      value: 71.647
    - type: map_at_1000
      value: 71.665
    - type: map_at_3
      value: 69.28
    - type: map_at_5
      value: 70.489
    - type: mrr_at_1
      value: 64.613
    - type: mrr_at_10
      value: 71.89
    - type: mrr_at_100
      value: 72.243
    - type: mrr_at_1000
      value: 72.259
    - type: mrr_at_3
      value: 70.138
    - type: mrr_at_5
      value: 71.232
    - type: ndcg_at_1
      value: 64.613
    - type: ndcg_at_10
      value: 75.005
    - type: ndcg_at_100
      value: 76.805
    - type: ndcg_at_1000
      value: 77.281
    - type: ndcg_at_3
      value: 71.234
    - type: ndcg_at_5
      value: 73.294
    - type: precision_at_1
      value: 64.613
    - type: precision_at_10
      value: 9.142
    - type: precision_at_100
      value: 1.004
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 26.781
    - type: precision_at_5
      value: 17.149
    - type: recall_at_1
      value: 62.517
    - type: recall_at_10
      value: 85.997
    - type: recall_at_100
      value: 94.18299999999999
    - type: recall_at_1000
      value: 97.911
    - type: recall_at_3
      value: 75.993
    - type: recall_at_5
      value: 80.88300000000001
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
      value: 59.27706792199058
    - type: f1
      value: 56.77545011902468
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
      value: 66.47948890383321
    - type: f1
      value: 66.4502180376861
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
      value: 54.2
    - type: map_at_10
      value: 59.858
    - type: map_at_100
      value: 60.46
    - type: map_at_1000
      value: 60.507
    - type: map_at_3
      value: 58.416999999999994
    - type: map_at_5
      value: 59.331999999999994
    - type: mrr_at_1
      value: 54.2
    - type: mrr_at_10
      value: 59.862
    - type: mrr_at_100
      value: 60.463
    - type: mrr_at_1000
      value: 60.51
    - type: mrr_at_3
      value: 58.416999999999994
    - type: mrr_at_5
      value: 59.352000000000004
    - type: ndcg_at_1
      value: 54.2
    - type: ndcg_at_10
      value: 62.643
    - type: ndcg_at_100
      value: 65.731
    - type: ndcg_at_1000
      value: 67.096
    - type: ndcg_at_3
      value: 59.727
    - type: ndcg_at_5
      value: 61.375
    - type: precision_at_1
      value: 54.2
    - type: precision_at_10
      value: 7.140000000000001
    - type: precision_at_100
      value: 0.8619999999999999
    - type: precision_at_1000
      value: 0.097
    - type: precision_at_3
      value: 21.166999999999998
    - type: precision_at_5
      value: 13.5
    - type: recall_at_1
      value: 54.2
    - type: recall_at_10
      value: 71.39999999999999
    - type: recall_at_100
      value: 86.2
    - type: recall_at_1000
      value: 97.2
    - type: recall_at_3
      value: 63.5
    - type: recall_at_5
      value: 67.5
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
      value: 68.19666666666666
    - type: f1
      value: 67.58581661416034
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
      value: 60.530590146182995
    - type: cos_sim_ap
      value: 63.53656091243922
    - type: cos_sim_f1
      value: 68.09929603556874
    - type: cos_sim_precision
      value: 52.45433789954338
    - type: cos_sim_recall
      value: 97.04329461457233
    - type: dot_accuracy
      value: 60.530590146182995
    - type: dot_ap
      value: 63.53660452157237
    - type: dot_f1
      value: 68.09929603556874
    - type: dot_precision
      value: 52.45433789954338
    - type: dot_recall
      value: 97.04329461457233
    - type: euclidean_accuracy
      value: 60.530590146182995
    - type: euclidean_ap
      value: 63.53678735855631
    - type: euclidean_f1
      value: 68.09929603556874
    - type: euclidean_precision
      value: 52.45433789954338
    - type: euclidean_recall
      value: 97.04329461457233
    - type: manhattan_accuracy
      value: 60.47644829453167
    - type: manhattan_ap
      value: 63.5622508250315
    - type: manhattan_f1
      value: 68.1650700073692
    - type: manhattan_precision
      value: 52.34861346915677
    - type: manhattan_recall
      value: 97.67687434002113
    - type: max_accuracy
      value: 60.530590146182995
    - type: max_ap
      value: 63.5622508250315
    - type: max_f1
      value: 68.1650700073692
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
      value: 89.13
    - type: ap
      value: 87.21879260137172
    - type: f1
      value: 89.12359325300508
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
      value: 12.035577637900758
    - type: cos_sim_spearman
      value: 12.76524190663864
    - type: euclidean_pearson
      value: 14.4012689427106
    - type: euclidean_spearman
      value: 12.765328992583608
    - type: manhattan_pearson
      value: 14.458505202938946
    - type: manhattan_spearman
      value: 12.763238700117896
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
      value: 34.809415339934006
    - type: cos_sim_spearman
      value: 36.96728615916954
    - type: euclidean_pearson
      value: 35.56113673772396
    - type: euclidean_spearman
      value: 36.96842963389308
    - type: manhattan_pearson
      value: 35.5447066178264
    - type: manhattan_spearman
      value: 36.97514513480951
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
      value: 66.39448692338551
    - type: cos_sim_spearman
      value: 66.72211526923901
    - type: euclidean_pearson
      value: 65.72981824553035
    - type: euclidean_spearman
      value: 66.72211526923901
    - type: manhattan_pearson
      value: 65.52315559414296
    - type: manhattan_spearman
      value: 66.61931702511545
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
      value: 76.73608064460915
    - type: cos_sim_spearman
      value: 76.51424826130031
    - type: euclidean_pearson
      value: 76.17930213372487
    - type: euclidean_spearman
      value: 76.51342756283478
    - type: manhattan_pearson
      value: 75.87085607319342
    - type: manhattan_spearman
      value: 76.22676341477134
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
      value: 65.38779931543048
    - type: mrr
      value: 74.79313763420059
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
      value: 25.131999999999998
    - type: map_at_10
      value: 69.131
    - type: map_at_100
      value: 72.943
    - type: map_at_1000
      value: 73.045
    - type: map_at_3
      value: 48.847
    - type: map_at_5
      value: 59.842
    - type: mrr_at_1
      value: 85.516
    - type: mrr_at_10
      value: 88.863
    - type: mrr_at_100
      value: 88.996
    - type: mrr_at_1000
      value: 89.00099999999999
    - type: mrr_at_3
      value: 88.277
    - type: mrr_at_5
      value: 88.64800000000001
    - type: ndcg_at_1
      value: 85.516
    - type: ndcg_at_10
      value: 78.122
    - type: ndcg_at_100
      value: 82.673
    - type: ndcg_at_1000
      value: 83.707
    - type: ndcg_at_3
      value: 80.274
    - type: ndcg_at_5
      value: 78.405
    - type: precision_at_1
      value: 85.516
    - type: precision_at_10
      value: 38.975
    - type: precision_at_100
      value: 4.833
    - type: precision_at_1000
      value: 0.509
    - type: precision_at_3
      value: 70.35
    - type: precision_at_5
      value: 58.638
    - type: recall_at_1
      value: 25.131999999999998
    - type: recall_at_10
      value: 76.848
    - type: recall_at_100
      value: 91.489
    - type: recall_at_1000
      value: 96.709
    - type: recall_at_3
      value: 50.824000000000005
    - type: recall_at_5
      value: 63.89
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
      value: 49.65
    - type: f1
      value: 47.66791473245483
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
      value: 63.78843565968542
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
      value: 55.14095244943176
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
      value: 53.800000000000004
    - type: map_at_10
      value: 63.312000000000005
    - type: map_at_100
      value: 63.93600000000001
    - type: map_at_1000
      value: 63.955
    - type: map_at_3
      value: 61.283
    - type: map_at_5
      value: 62.553000000000004
    - type: mrr_at_1
      value: 53.800000000000004
    - type: mrr_at_10
      value: 63.312000000000005
    - type: mrr_at_100
      value: 63.93600000000001
    - type: mrr_at_1000
      value: 63.955
    - type: mrr_at_3
      value: 61.283
    - type: mrr_at_5
      value: 62.553000000000004
    - type: ndcg_at_1
      value: 53.800000000000004
    - type: ndcg_at_10
      value: 67.693
    - type: ndcg_at_100
      value: 70.552
    - type: ndcg_at_1000
      value: 71.06099999999999
    - type: ndcg_at_3
      value: 63.632
    - type: ndcg_at_5
      value: 65.90899999999999
    - type: precision_at_1
      value: 53.800000000000004
    - type: precision_at_10
      value: 8.129999999999999
    - type: precision_at_100
      value: 0.943
    - type: precision_at_1000
      value: 0.098
    - type: precision_at_3
      value: 23.467
    - type: precision_at_5
      value: 15.18
    - type: recall_at_1
      value: 53.800000000000004
    - type: recall_at_10
      value: 81.3
    - type: recall_at_100
      value: 94.3
    - type: recall_at_1000
      value: 98.3
    - type: recall_at_3
      value: 70.39999999999999
    - type: recall_at_5
      value: 75.9
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
      value: 84.96000000000001
    - type: ap
      value: 66.89917287702019
    - type: f1
      value: 83.0239988458119
---

# gte-small-zh

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

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small-zh")
model = AutoModel.from_pretrained("thenlper/gte-small-zh")

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

model = SentenceTransformer('thenlper/gte-small-zh')
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
