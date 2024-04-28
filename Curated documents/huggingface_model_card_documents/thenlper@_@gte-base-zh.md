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
- name: gte-base-zh
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
      value: 44.45621572456527
    - type: cos_sim_spearman
      value: 49.06500895667604
    - type: euclidean_pearson
      value: 47.55002064096053
    - type: euclidean_spearman
      value: 49.06500895667604
    - type: manhattan_pearson
      value: 47.429900262366715
    - type: manhattan_spearman
      value: 48.95704890278774
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
      value: 44.31699346653116
    - type: cos_sim_spearman
      value: 50.83133156721432
    - type: euclidean_pearson
      value: 51.36086517946001
    - type: euclidean_spearman
      value: 50.83132818894256
    - type: manhattan_pearson
      value: 51.255926461574084
    - type: manhattan_spearman
      value: 50.73460147395406
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
      value: 45.818000000000005
    - type: f1
      value: 43.998253644678144
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
      value: 63.47477451918581
    - type: cos_sim_spearman
      value: 65.49832607366159
    - type: euclidean_pearson
      value: 64.11399760832107
    - type: euclidean_spearman
      value: 65.49832260877398
    - type: manhattan_pearson
      value: 64.02541311484639
    - type: manhattan_spearman
      value: 65.42436057501452
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
      value: 42.58046835435111
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
      value: 40.42134173217685
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
      value: 86.79079943923792
    - type: mrr
      value: 88.81341269841269
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
      value: 87.20186031249037
    - type: mrr
      value: 89.46551587301587
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
      value: 25.098
    - type: map_at_10
      value: 37.759
    - type: map_at_100
      value: 39.693
    - type: map_at_1000
      value: 39.804
    - type: map_at_3
      value: 33.477000000000004
    - type: map_at_5
      value: 35.839
    - type: mrr_at_1
      value: 38.06
    - type: mrr_at_10
      value: 46.302
    - type: mrr_at_100
      value: 47.370000000000005
    - type: mrr_at_1000
      value: 47.412
    - type: mrr_at_3
      value: 43.702999999999996
    - type: mrr_at_5
      value: 45.213
    - type: ndcg_at_1
      value: 38.06
    - type: ndcg_at_10
      value: 44.375
    - type: ndcg_at_100
      value: 51.849999999999994
    - type: ndcg_at_1000
      value: 53.725
    - type: ndcg_at_3
      value: 38.97
    - type: ndcg_at_5
      value: 41.193000000000005
    - type: precision_at_1
      value: 38.06
    - type: precision_at_10
      value: 9.934999999999999
    - type: precision_at_100
      value: 1.599
    - type: precision_at_1000
      value: 0.183
    - type: precision_at_3
      value: 22.072
    - type: precision_at_5
      value: 16.089000000000002
    - type: recall_at_1
      value: 25.098
    - type: recall_at_10
      value: 55.264
    - type: recall_at_100
      value: 85.939
    - type: recall_at_1000
      value: 98.44800000000001
    - type: recall_at_3
      value: 39.122
    - type: recall_at_5
      value: 45.948
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
      value: 78.02766085387853
    - type: cos_sim_ap
      value: 85.59982802559004
    - type: cos_sim_f1
      value: 79.57103418984921
    - type: cos_sim_precision
      value: 72.88465279128575
    - type: cos_sim_recall
      value: 87.60813654430676
    - type: dot_accuracy
      value: 78.02766085387853
    - type: dot_ap
      value: 85.59604477360719
    - type: dot_f1
      value: 79.57103418984921
    - type: dot_precision
      value: 72.88465279128575
    - type: dot_recall
      value: 87.60813654430676
    - type: euclidean_accuracy
      value: 78.02766085387853
    - type: euclidean_ap
      value: 85.59982802559004
    - type: euclidean_f1
      value: 79.57103418984921
    - type: euclidean_precision
      value: 72.88465279128575
    - type: euclidean_recall
      value: 87.60813654430676
    - type: manhattan_accuracy
      value: 77.9795550210463
    - type: manhattan_ap
      value: 85.58042267497707
    - type: manhattan_f1
      value: 79.40344001741781
    - type: manhattan_precision
      value: 74.29211652067632
    - type: manhattan_recall
      value: 85.27004909983633
    - type: max_accuracy
      value: 78.02766085387853
    - type: max_ap
      value: 85.59982802559004
    - type: max_f1
      value: 79.57103418984921
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
      value: 62.144
    - type: map_at_10
      value: 71.589
    - type: map_at_100
      value: 72.066
    - type: map_at_1000
      value: 72.075
    - type: map_at_3
      value: 69.916
    - type: map_at_5
      value: 70.806
    - type: mrr_at_1
      value: 62.275999999999996
    - type: mrr_at_10
      value: 71.57
    - type: mrr_at_100
      value: 72.048
    - type: mrr_at_1000
      value: 72.057
    - type: mrr_at_3
      value: 69.89800000000001
    - type: mrr_at_5
      value: 70.84700000000001
    - type: ndcg_at_1
      value: 62.381
    - type: ndcg_at_10
      value: 75.74
    - type: ndcg_at_100
      value: 77.827
    - type: ndcg_at_1000
      value: 78.044
    - type: ndcg_at_3
      value: 72.307
    - type: ndcg_at_5
      value: 73.91499999999999
    - type: precision_at_1
      value: 62.381
    - type: precision_at_10
      value: 8.946
    - type: precision_at_100
      value: 0.988
    - type: precision_at_1000
      value: 0.101
    - type: precision_at_3
      value: 26.554
    - type: precision_at_5
      value: 16.733
    - type: recall_at_1
      value: 62.144
    - type: recall_at_10
      value: 88.567
    - type: recall_at_100
      value: 97.84
    - type: recall_at_1000
      value: 99.473
    - type: recall_at_3
      value: 79.083
    - type: recall_at_5
      value: 83.035
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
      value: 24.665
    - type: map_at_10
      value: 74.91600000000001
    - type: map_at_100
      value: 77.981
    - type: map_at_1000
      value: 78.032
    - type: map_at_3
      value: 51.015
    - type: map_at_5
      value: 64.681
    - type: mrr_at_1
      value: 86.5
    - type: mrr_at_10
      value: 90.78399999999999
    - type: mrr_at_100
      value: 90.859
    - type: mrr_at_1000
      value: 90.863
    - type: mrr_at_3
      value: 90.375
    - type: mrr_at_5
      value: 90.66199999999999
    - type: ndcg_at_1
      value: 86.5
    - type: ndcg_at_10
      value: 83.635
    - type: ndcg_at_100
      value: 86.926
    - type: ndcg_at_1000
      value: 87.425
    - type: ndcg_at_3
      value: 81.28999999999999
    - type: ndcg_at_5
      value: 80.549
    - type: precision_at_1
      value: 86.5
    - type: precision_at_10
      value: 40.544999999999995
    - type: precision_at_100
      value: 4.748
    - type: precision_at_1000
      value: 0.48700000000000004
    - type: precision_at_3
      value: 72.68299999999999
    - type: precision_at_5
      value: 61.86000000000001
    - type: recall_at_1
      value: 24.665
    - type: recall_at_10
      value: 85.72
    - type: recall_at_100
      value: 96.116
    - type: recall_at_1000
      value: 98.772
    - type: recall_at_3
      value: 53.705999999999996
    - type: recall_at_5
      value: 70.42699999999999
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
      value: 54.0
    - type: map_at_10
      value: 64.449
    - type: map_at_100
      value: 64.937
    - type: map_at_1000
      value: 64.946
    - type: map_at_3
      value: 61.85000000000001
    - type: map_at_5
      value: 63.525
    - type: mrr_at_1
      value: 54.0
    - type: mrr_at_10
      value: 64.449
    - type: mrr_at_100
      value: 64.937
    - type: mrr_at_1000
      value: 64.946
    - type: mrr_at_3
      value: 61.85000000000001
    - type: mrr_at_5
      value: 63.525
    - type: ndcg_at_1
      value: 54.0
    - type: ndcg_at_10
      value: 69.56400000000001
    - type: ndcg_at_100
      value: 71.78999999999999
    - type: ndcg_at_1000
      value: 72.021
    - type: ndcg_at_3
      value: 64.334
    - type: ndcg_at_5
      value: 67.368
    - type: precision_at_1
      value: 54.0
    - type: precision_at_10
      value: 8.559999999999999
    - type: precision_at_100
      value: 0.9570000000000001
    - type: precision_at_1000
      value: 0.098
    - type: precision_at_3
      value: 23.833
    - type: precision_at_5
      value: 15.78
    - type: recall_at_1
      value: 54.0
    - type: recall_at_10
      value: 85.6
    - type: recall_at_100
      value: 95.7
    - type: recall_at_1000
      value: 97.5
    - type: recall_at_3
      value: 71.5
    - type: recall_at_5
      value: 78.9
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
      value: 48.61869949980762
    - type: f1
      value: 36.49337336098832
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
      value: 85.94746716697938
    - type: ap
      value: 53.75927589310753
    - type: f1
      value: 80.53821597736138
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
      value: 68.77445518082875
    - type: cos_sim_spearman
      value: 74.05909185405268
    - type: euclidean_pearson
      value: 72.92870557009725
    - type: euclidean_spearman
      value: 74.05909628639644
    - type: manhattan_pearson
      value: 72.92072580598351
    - type: manhattan_spearman
      value: 74.0304390211741
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
      value: 27.643607073221975
    - type: mrr
      value: 26.646825396825395
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
      value: 65.10000000000001
    - type: map_at_10
      value: 74.014
    - type: map_at_100
      value: 74.372
    - type: map_at_1000
      value: 74.385
    - type: map_at_3
      value: 72.179
    - type: map_at_5
      value: 73.37700000000001
    - type: mrr_at_1
      value: 67.364
    - type: mrr_at_10
      value: 74.68
    - type: mrr_at_100
      value: 74.992
    - type: mrr_at_1000
      value: 75.003
    - type: mrr_at_3
      value: 73.054
    - type: mrr_at_5
      value: 74.126
    - type: ndcg_at_1
      value: 67.364
    - type: ndcg_at_10
      value: 77.704
    - type: ndcg_at_100
      value: 79.29899999999999
    - type: ndcg_at_1000
      value: 79.637
    - type: ndcg_at_3
      value: 74.232
    - type: ndcg_at_5
      value: 76.264
    - type: precision_at_1
      value: 67.364
    - type: precision_at_10
      value: 9.397
    - type: precision_at_100
      value: 1.019
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_3
      value: 27.942
    - type: precision_at_5
      value: 17.837
    - type: recall_at_1
      value: 65.10000000000001
    - type: recall_at_10
      value: 88.416
    - type: recall_at_100
      value: 95.61
    - type: recall_at_1000
      value: 98.261
    - type: recall_at_3
      value: 79.28
    - type: recall_at_5
      value: 84.108
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
      value: 73.315400134499
    - type: f1
      value: 70.81060697693198
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
      value: 76.78883658372563
    - type: f1
      value: 76.21512438791976
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
      value: 55.300000000000004
    - type: map_at_10
      value: 61.879
    - type: map_at_100
      value: 62.434
    - type: map_at_1000
      value: 62.476
    - type: map_at_3
      value: 60.417
    - type: map_at_5
      value: 61.297000000000004
    - type: mrr_at_1
      value: 55.400000000000006
    - type: mrr_at_10
      value: 61.92100000000001
    - type: mrr_at_100
      value: 62.476
    - type: mrr_at_1000
      value: 62.517999999999994
    - type: mrr_at_3
      value: 60.483
    - type: mrr_at_5
      value: 61.338
    - type: ndcg_at_1
      value: 55.300000000000004
    - type: ndcg_at_10
      value: 64.937
    - type: ndcg_at_100
      value: 67.848
    - type: ndcg_at_1000
      value: 68.996
    - type: ndcg_at_3
      value: 61.939
    - type: ndcg_at_5
      value: 63.556999999999995
    - type: precision_at_1
      value: 55.300000000000004
    - type: precision_at_10
      value: 7.449999999999999
    - type: precision_at_100
      value: 0.886
    - type: precision_at_1000
      value: 0.098
    - type: precision_at_3
      value: 22.1
    - type: precision_at_5
      value: 14.06
    - type: recall_at_1
      value: 55.300000000000004
    - type: recall_at_10
      value: 74.5
    - type: recall_at_100
      value: 88.6
    - type: recall_at_1000
      value: 97.7
    - type: recall_at_3
      value: 66.3
    - type: recall_at_5
      value: 70.3
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
      value: 75.79
    - type: f1
      value: 75.58944709087194
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
      value: 71.5755278830536
    - type: cos_sim_ap
      value: 75.27777388526098
    - type: cos_sim_f1
      value: 75.04604051565377
    - type: cos_sim_precision
      value: 66.53061224489795
    - type: cos_sim_recall
      value: 86.06124604012672
    - type: dot_accuracy
      value: 71.5755278830536
    - type: dot_ap
      value: 75.27765883143745
    - type: dot_f1
      value: 75.04604051565377
    - type: dot_precision
      value: 66.53061224489795
    - type: dot_recall
      value: 86.06124604012672
    - type: euclidean_accuracy
      value: 71.5755278830536
    - type: euclidean_ap
      value: 75.27762982049899
    - type: euclidean_f1
      value: 75.04604051565377
    - type: euclidean_precision
      value: 66.53061224489795
    - type: euclidean_recall
      value: 86.06124604012672
    - type: manhattan_accuracy
      value: 71.41310232809963
    - type: manhattan_ap
      value: 75.11908556317425
    - type: manhattan_f1
      value: 75.0118091639112
    - type: manhattan_precision
      value: 67.86324786324786
    - type: manhattan_recall
      value: 83.84371700105596
    - type: max_accuracy
      value: 71.5755278830536
    - type: max_ap
      value: 75.27777388526098
    - type: max_f1
      value: 75.04604051565377
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
      value: 93.36
    - type: ap
      value: 91.66871784150999
    - type: f1
      value: 93.35216314755989
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
      value: 24.21926662784366
    - type: cos_sim_spearman
      value: 27.969680921064644
    - type: euclidean_pearson
      value: 28.75506415195721
    - type: euclidean_spearman
      value: 27.969593815056058
    - type: manhattan_pearson
      value: 28.90608040712011
    - type: manhattan_spearman
      value: 28.07097299964309
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
      value: 33.4112661812038
    - type: cos_sim_spearman
      value: 35.192765228905174
    - type: euclidean_pearson
      value: 33.57803958232971
    - type: euclidean_spearman
      value: 35.19270413260232
    - type: manhattan_pearson
      value: 33.75933288702631
    - type: manhattan_spearman
      value: 35.362780488430126
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
      value: 62.178764479940206
    - type: cos_sim_spearman
      value: 63.644049344272155
    - type: euclidean_pearson
      value: 61.97852518030118
    - type: euclidean_spearman
      value: 63.644049344272155
    - type: manhattan_pearson
      value: 62.3931275533103
    - type: manhattan_spearman
      value: 63.68720814152202
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
      value: 81.09847341753118
    - type: cos_sim_spearman
      value: 81.46211495319093
    - type: euclidean_pearson
      value: 80.97905808856734
    - type: euclidean_spearman
      value: 81.46177732221445
    - type: manhattan_pearson
      value: 80.8737913286308
    - type: manhattan_spearman
      value: 81.41142532907402
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
      value: 66.36295416100998
    - type: mrr
      value: 76.42041058129412
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
      value: 26.898
    - type: map_at_10
      value: 75.089
    - type: map_at_100
      value: 78.786
    - type: map_at_1000
      value: 78.86
    - type: map_at_3
      value: 52.881
    - type: map_at_5
      value: 64.881
    - type: mrr_at_1
      value: 88.984
    - type: mrr_at_10
      value: 91.681
    - type: mrr_at_100
      value: 91.77300000000001
    - type: mrr_at_1000
      value: 91.777
    - type: mrr_at_3
      value: 91.205
    - type: mrr_at_5
      value: 91.486
    - type: ndcg_at_1
      value: 88.984
    - type: ndcg_at_10
      value: 83.083
    - type: ndcg_at_100
      value: 86.955
    - type: ndcg_at_1000
      value: 87.665
    - type: ndcg_at_3
      value: 84.661
    - type: ndcg_at_5
      value: 83.084
    - type: precision_at_1
      value: 88.984
    - type: precision_at_10
      value: 41.311
    - type: precision_at_100
      value: 4.978
    - type: precision_at_1000
      value: 0.515
    - type: precision_at_3
      value: 74.074
    - type: precision_at_5
      value: 61.956999999999994
    - type: recall_at_1
      value: 26.898
    - type: recall_at_10
      value: 82.03200000000001
    - type: recall_at_100
      value: 94.593
    - type: recall_at_1000
      value: 98.188
    - type: recall_at_3
      value: 54.647999999999996
    - type: recall_at_5
      value: 68.394
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
      value: 53.648999999999994
    - type: f1
      value: 51.87788185753318
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
      value: 68.81293224496076
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
      value: 63.60504270553153
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
      value: 69.89
    - type: map_at_100
      value: 70.261
    - type: map_at_1000
      value: 70.27
    - type: map_at_3
      value: 67.93299999999999
    - type: map_at_5
      value: 69.10300000000001
    - type: mrr_at_1
      value: 59.3
    - type: mrr_at_10
      value: 69.89
    - type: mrr_at_100
      value: 70.261
    - type: mrr_at_1000
      value: 70.27
    - type: mrr_at_3
      value: 67.93299999999999
    - type: mrr_at_5
      value: 69.10300000000001
    - type: ndcg_at_1
      value: 59.3
    - type: ndcg_at_10
      value: 74.67099999999999
    - type: ndcg_at_100
      value: 76.371
    - type: ndcg_at_1000
      value: 76.644
    - type: ndcg_at_3
      value: 70.678
    - type: ndcg_at_5
      value: 72.783
    - type: precision_at_1
      value: 59.3
    - type: precision_at_10
      value: 8.95
    - type: precision_at_100
      value: 0.972
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 26.200000000000003
    - type: precision_at_5
      value: 16.74
    - type: recall_at_1
      value: 59.3
    - type: recall_at_10
      value: 89.5
    - type: recall_at_100
      value: 97.2
    - type: recall_at_1000
      value: 99.4
    - type: recall_at_3
      value: 78.60000000000001
    - type: recall_at_5
      value: 83.7
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
      value: 88.07000000000001
    - type: ap
      value: 72.68881791758656
    - type: f1
      value: 86.647906274628
---

# gte-base-zh

General Text Embeddings (GTE) model. [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)

The GTE models are trained by Alibaba DAMO Academy. They are mainly based on the BERT framework and currently offer different sizes of models for both Chinese and English Languages. The GTE models are trained on a large-scale corpus of relevance text pairs, covering a wide range of domains and scenarios. This enables the GTE models to be applied to various downstream tasks of text embeddings, including **information retrieval**, **semantic textual similarity**, **text reranking**, etc.

## Model List

| Models | Language | Max Sequence Length | Dimension | Model Size |
|:-----: | :-----: |:-----: |:-----: |:-----: |
|[GTE-large-zh](https://huggingface.co/thenlper/gte-large-zh) | Chinese | 512 | 1024 | 0.67GB |
|[GTE-base-zh](https://huggingface.co/thenlper/gte-base-zh) | Chinese | 512 | 1024 | 0.67GB |
|[GTE-small-zh](https://huggingface.co/thenlper/gte-small-zh) | Chinese | 512 | 1024 | 0.67GB |
|[GTE-large](https://huggingface.co/thenlper/gte-large) | English | 512 | 1024 | 0.67GB |
|[GTE-base](https://huggingface.co/thenlper/gte-base) | English | 512 | 1024 | 0.67GB |
|[GTE-small](https://huggingface.co/thenlper/gte-small) | English | 512 | 1024 | 0.67GB |


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
| gte-small-zh         | 0.1           | 512                  | 512              | 60.08                 | 64.49                                | 48.95                          | 69.99                                   | 66.21                          | 65.50                        | 49.72                    |
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

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base-zh")
model = AutoModel.from_pretrained("thenlper/gte-base-zh")

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

sentences = ['中国的首都是哪里', '中国的首都是北京']

model = SentenceTransformer('thenlper/gte-base-zh')
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
