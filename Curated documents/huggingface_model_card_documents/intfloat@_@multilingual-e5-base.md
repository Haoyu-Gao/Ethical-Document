---
language:
- multilingual
- af
- am
- ar
- as
- az
- be
- bg
- bn
- br
- bs
- ca
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- he
- hi
- hr
- hu
- hy
- id
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lo
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- my
- ne
- nl
- 'no'
- om
- or
- pa
- pl
- ps
- pt
- ro
- ru
- sa
- sd
- si
- sk
- sl
- so
- sq
- sr
- su
- sv
- sw
- ta
- te
- th
- tl
- tr
- ug
- uk
- ur
- uz
- vi
- xh
- yi
- zh
license: mit
tags:
- mteb
- Sentence Transformers
- sentence-similarity
- sentence-transformers
model-index:
- name: multilingual-e5-base
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
      value: 78.97014925373135
    - type: ap
      value: 43.69351129103008
    - type: f1
      value: 73.38075030070492
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (de)
      type: mteb/amazon_counterfactual
      config: de
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 71.7237687366167
    - type: ap
      value: 82.22089859962671
    - type: f1
      value: 69.95532758884401
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (en-ext)
      type: mteb/amazon_counterfactual
      config: en-ext
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 79.65517241379312
    - type: ap
      value: 28.507918657094738
    - type: f1
      value: 66.84516013726119
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonCounterfactualClassification (ja)
      type: mteb/amazon_counterfactual
      config: ja
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 73.32976445396146
    - type: ap
      value: 20.720481637566014
    - type: f1
      value: 59.78002763416003
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
      value: 90.63775
    - type: ap
      value: 87.22277903861716
    - type: f1
      value: 90.60378636386807
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
      value: 44.546
    - type: f1
      value: 44.05666638370923
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (de)
      type: mteb/amazon_reviews_multi
      config: de
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 41.828
    - type: f1
      value: 41.2710255644252
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (es)
      type: mteb/amazon_reviews_multi
      config: es
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 40.534
    - type: f1
      value: 39.820743174270326
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (fr)
      type: mteb/amazon_reviews_multi
      config: fr
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 39.684
    - type: f1
      value: 39.11052682815307
  - task:
      type: Classification
    dataset:
      name: MTEB AmazonReviewsClassification (ja)
      type: mteb/amazon_reviews_multi
      config: ja
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 37.436
    - type: f1
      value: 37.07082931930871
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
      value: 37.226000000000006
    - type: f1
      value: 36.65372077739185
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
      value: 22.831000000000003
    - type: map_at_10
      value: 36.42
    - type: map_at_100
      value: 37.699
    - type: map_at_1000
      value: 37.724000000000004
    - type: map_at_3
      value: 32.207
    - type: map_at_5
      value: 34.312
    - type: mrr_at_1
      value: 23.257
    - type: mrr_at_10
      value: 36.574
    - type: mrr_at_100
      value: 37.854
    - type: mrr_at_1000
      value: 37.878
    - type: mrr_at_3
      value: 32.385000000000005
    - type: mrr_at_5
      value: 34.48
    - type: ndcg_at_1
      value: 22.831000000000003
    - type: ndcg_at_10
      value: 44.230000000000004
    - type: ndcg_at_100
      value: 49.974000000000004
    - type: ndcg_at_1000
      value: 50.522999999999996
    - type: ndcg_at_3
      value: 35.363
    - type: ndcg_at_5
      value: 39.164
    - type: precision_at_1
      value: 22.831000000000003
    - type: precision_at_10
      value: 6.935
    - type: precision_at_100
      value: 0.9520000000000001
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 14.841
    - type: precision_at_5
      value: 10.754
    - type: recall_at_1
      value: 22.831000000000003
    - type: recall_at_10
      value: 69.346
    - type: recall_at_100
      value: 95.235
    - type: recall_at_1000
      value: 99.36
    - type: recall_at_3
      value: 44.523
    - type: recall_at_5
      value: 53.769999999999996
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
      value: 40.27789869854063
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
      value: 35.41979463347428
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
      value: 58.22752045109304
    - type: mrr
      value: 71.51112430198303
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
      value: 84.71147646622866
    - type: cos_sim_spearman
      value: 85.059167046486
    - type: euclidean_pearson
      value: 75.88421613600647
    - type: euclidean_spearman
      value: 75.12821787150585
    - type: manhattan_pearson
      value: 75.22005646957604
    - type: manhattan_spearman
      value: 74.42880434453272
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (de-en)
      type: mteb/bucc-bitext-mining
      config: de-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 99.23799582463465
    - type: f1
      value: 99.12665274878218
    - type: precision
      value: 99.07098121085595
    - type: recall
      value: 99.23799582463465
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (fr-en)
      type: mteb/bucc-bitext-mining
      config: fr-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 97.88685890380806
    - type: f1
      value: 97.59336708489249
    - type: precision
      value: 97.44662117543473
    - type: recall
      value: 97.88685890380806
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (ru-en)
      type: mteb/bucc-bitext-mining
      config: ru-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 97.47142362313821
    - type: f1
      value: 97.1989377670015
    - type: precision
      value: 97.06384944001847
    - type: recall
      value: 97.47142362313821
  - task:
      type: BitextMining
    dataset:
      name: MTEB BUCC (zh-en)
      type: mteb/bucc-bitext-mining
      config: zh-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 98.4728804634018
    - type: f1
      value: 98.2973494821836
    - type: precision
      value: 98.2095839915745
    - type: recall
      value: 98.4728804634018
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
      value: 82.74025974025975
    - type: f1
      value: 82.67420447730439
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
      value: 35.0380848063507
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
      value: 29.45956405670166
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
      value: 32.122
    - type: map_at_10
      value: 42.03
    - type: map_at_100
      value: 43.364000000000004
    - type: map_at_1000
      value: 43.474000000000004
    - type: map_at_3
      value: 38.804
    - type: map_at_5
      value: 40.585
    - type: mrr_at_1
      value: 39.914
    - type: mrr_at_10
      value: 48.227
    - type: mrr_at_100
      value: 49.018
    - type: mrr_at_1000
      value: 49.064
    - type: mrr_at_3
      value: 45.994
    - type: mrr_at_5
      value: 47.396
    - type: ndcg_at_1
      value: 39.914
    - type: ndcg_at_10
      value: 47.825
    - type: ndcg_at_100
      value: 52.852
    - type: ndcg_at_1000
      value: 54.891
    - type: ndcg_at_3
      value: 43.517
    - type: ndcg_at_5
      value: 45.493
    - type: precision_at_1
      value: 39.914
    - type: precision_at_10
      value: 8.956
    - type: precision_at_100
      value: 1.388
    - type: precision_at_1000
      value: 0.182
    - type: precision_at_3
      value: 20.791999999999998
    - type: precision_at_5
      value: 14.821000000000002
    - type: recall_at_1
      value: 32.122
    - type: recall_at_10
      value: 58.294999999999995
    - type: recall_at_100
      value: 79.726
    - type: recall_at_1000
      value: 93.099
    - type: recall_at_3
      value: 45.017
    - type: recall_at_5
      value: 51.002
    - type: map_at_1
      value: 29.677999999999997
    - type: map_at_10
      value: 38.684000000000005
    - type: map_at_100
      value: 39.812999999999995
    - type: map_at_1000
      value: 39.945
    - type: map_at_3
      value: 35.831
    - type: map_at_5
      value: 37.446
    - type: mrr_at_1
      value: 37.771
    - type: mrr_at_10
      value: 44.936
    - type: mrr_at_100
      value: 45.583
    - type: mrr_at_1000
      value: 45.634
    - type: mrr_at_3
      value: 42.771
    - type: mrr_at_5
      value: 43.994
    - type: ndcg_at_1
      value: 37.771
    - type: ndcg_at_10
      value: 44.059
    - type: ndcg_at_100
      value: 48.192
    - type: ndcg_at_1000
      value: 50.375
    - type: ndcg_at_3
      value: 40.172000000000004
    - type: ndcg_at_5
      value: 41.899
    - type: precision_at_1
      value: 37.771
    - type: precision_at_10
      value: 8.286999999999999
    - type: precision_at_100
      value: 1.322
    - type: precision_at_1000
      value: 0.178
    - type: precision_at_3
      value: 19.406000000000002
    - type: precision_at_5
      value: 13.745
    - type: recall_at_1
      value: 29.677999999999997
    - type: recall_at_10
      value: 53.071
    - type: recall_at_100
      value: 70.812
    - type: recall_at_1000
      value: 84.841
    - type: recall_at_3
      value: 41.016000000000005
    - type: recall_at_5
      value: 46.22
    - type: map_at_1
      value: 42.675000000000004
    - type: map_at_10
      value: 53.93599999999999
    - type: map_at_100
      value: 54.806999999999995
    - type: map_at_1000
      value: 54.867
    - type: map_at_3
      value: 50.934000000000005
    - type: map_at_5
      value: 52.583
    - type: mrr_at_1
      value: 48.339
    - type: mrr_at_10
      value: 57.265
    - type: mrr_at_100
      value: 57.873
    - type: mrr_at_1000
      value: 57.906
    - type: mrr_at_3
      value: 55.193000000000005
    - type: mrr_at_5
      value: 56.303000000000004
    - type: ndcg_at_1
      value: 48.339
    - type: ndcg_at_10
      value: 59.19799999999999
    - type: ndcg_at_100
      value: 62.743
    - type: ndcg_at_1000
      value: 63.99399999999999
    - type: ndcg_at_3
      value: 54.367
    - type: ndcg_at_5
      value: 56.548
    - type: precision_at_1
      value: 48.339
    - type: precision_at_10
      value: 9.216000000000001
    - type: precision_at_100
      value: 1.1809999999999998
    - type: precision_at_1000
      value: 0.134
    - type: precision_at_3
      value: 23.72
    - type: precision_at_5
      value: 16.025
    - type: recall_at_1
      value: 42.675000000000004
    - type: recall_at_10
      value: 71.437
    - type: recall_at_100
      value: 86.803
    - type: recall_at_1000
      value: 95.581
    - type: recall_at_3
      value: 58.434
    - type: recall_at_5
      value: 63.754
    - type: map_at_1
      value: 23.518
    - type: map_at_10
      value: 30.648999999999997
    - type: map_at_100
      value: 31.508999999999997
    - type: map_at_1000
      value: 31.604
    - type: map_at_3
      value: 28.247
    - type: map_at_5
      value: 29.65
    - type: mrr_at_1
      value: 25.650000000000002
    - type: mrr_at_10
      value: 32.771
    - type: mrr_at_100
      value: 33.554
    - type: mrr_at_1000
      value: 33.629999999999995
    - type: mrr_at_3
      value: 30.433
    - type: mrr_at_5
      value: 31.812
    - type: ndcg_at_1
      value: 25.650000000000002
    - type: ndcg_at_10
      value: 34.929
    - type: ndcg_at_100
      value: 39.382
    - type: ndcg_at_1000
      value: 41.913
    - type: ndcg_at_3
      value: 30.292
    - type: ndcg_at_5
      value: 32.629999999999995
    - type: precision_at_1
      value: 25.650000000000002
    - type: precision_at_10
      value: 5.311
    - type: precision_at_100
      value: 0.792
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_3
      value: 12.58
    - type: precision_at_5
      value: 8.994
    - type: recall_at_1
      value: 23.518
    - type: recall_at_10
      value: 46.19
    - type: recall_at_100
      value: 67.123
    - type: recall_at_1000
      value: 86.442
    - type: recall_at_3
      value: 33.678000000000004
    - type: recall_at_5
      value: 39.244
    - type: map_at_1
      value: 15.891
    - type: map_at_10
      value: 22.464000000000002
    - type: map_at_100
      value: 23.483
    - type: map_at_1000
      value: 23.613
    - type: map_at_3
      value: 20.080000000000002
    - type: map_at_5
      value: 21.526
    - type: mrr_at_1
      value: 20.025000000000002
    - type: mrr_at_10
      value: 26.712999999999997
    - type: mrr_at_100
      value: 27.650000000000002
    - type: mrr_at_1000
      value: 27.737000000000002
    - type: mrr_at_3
      value: 24.274
    - type: mrr_at_5
      value: 25.711000000000002
    - type: ndcg_at_1
      value: 20.025000000000002
    - type: ndcg_at_10
      value: 27.028999999999996
    - type: ndcg_at_100
      value: 32.064
    - type: ndcg_at_1000
      value: 35.188
    - type: ndcg_at_3
      value: 22.512999999999998
    - type: ndcg_at_5
      value: 24.89
    - type: precision_at_1
      value: 20.025000000000002
    - type: precision_at_10
      value: 4.776
    - type: precision_at_100
      value: 0.8500000000000001
    - type: precision_at_1000
      value: 0.125
    - type: precision_at_3
      value: 10.531
    - type: precision_at_5
      value: 7.811
    - type: recall_at_1
      value: 15.891
    - type: recall_at_10
      value: 37.261
    - type: recall_at_100
      value: 59.12
    - type: recall_at_1000
      value: 81.356
    - type: recall_at_3
      value: 24.741
    - type: recall_at_5
      value: 30.753999999999998
    - type: map_at_1
      value: 27.544
    - type: map_at_10
      value: 36.283
    - type: map_at_100
      value: 37.467
    - type: map_at_1000
      value: 37.574000000000005
    - type: map_at_3
      value: 33.528999999999996
    - type: map_at_5
      value: 35.028999999999996
    - type: mrr_at_1
      value: 34.166999999999994
    - type: mrr_at_10
      value: 41.866
    - type: mrr_at_100
      value: 42.666
    - type: mrr_at_1000
      value: 42.716
    - type: mrr_at_3
      value: 39.541
    - type: mrr_at_5
      value: 40.768
    - type: ndcg_at_1
      value: 34.166999999999994
    - type: ndcg_at_10
      value: 41.577
    - type: ndcg_at_100
      value: 46.687
    - type: ndcg_at_1000
      value: 48.967
    - type: ndcg_at_3
      value: 37.177
    - type: ndcg_at_5
      value: 39.097
    - type: precision_at_1
      value: 34.166999999999994
    - type: precision_at_10
      value: 7.420999999999999
    - type: precision_at_100
      value: 1.165
    - type: precision_at_1000
      value: 0.154
    - type: precision_at_3
      value: 17.291999999999998
    - type: precision_at_5
      value: 12.166
    - type: recall_at_1
      value: 27.544
    - type: recall_at_10
      value: 51.99399999999999
    - type: recall_at_100
      value: 73.738
    - type: recall_at_1000
      value: 89.33
    - type: recall_at_3
      value: 39.179
    - type: recall_at_5
      value: 44.385999999999996
    - type: map_at_1
      value: 26.661
    - type: map_at_10
      value: 35.475
    - type: map_at_100
      value: 36.626999999999995
    - type: map_at_1000
      value: 36.741
    - type: map_at_3
      value: 32.818000000000005
    - type: map_at_5
      value: 34.397
    - type: mrr_at_1
      value: 32.647999999999996
    - type: mrr_at_10
      value: 40.784
    - type: mrr_at_100
      value: 41.602
    - type: mrr_at_1000
      value: 41.661
    - type: mrr_at_3
      value: 38.68
    - type: mrr_at_5
      value: 39.838
    - type: ndcg_at_1
      value: 32.647999999999996
    - type: ndcg_at_10
      value: 40.697
    - type: ndcg_at_100
      value: 45.799
    - type: ndcg_at_1000
      value: 48.235
    - type: ndcg_at_3
      value: 36.516
    - type: ndcg_at_5
      value: 38.515
    - type: precision_at_1
      value: 32.647999999999996
    - type: precision_at_10
      value: 7.202999999999999
    - type: precision_at_100
      value: 1.1360000000000001
    - type: precision_at_1000
      value: 0.151
    - type: precision_at_3
      value: 17.314
    - type: precision_at_5
      value: 12.145999999999999
    - type: recall_at_1
      value: 26.661
    - type: recall_at_10
      value: 50.995000000000005
    - type: recall_at_100
      value: 73.065
    - type: recall_at_1000
      value: 89.781
    - type: recall_at_3
      value: 39.073
    - type: recall_at_5
      value: 44.395
    - type: map_at_1
      value: 25.946583333333333
    - type: map_at_10
      value: 33.79725
    - type: map_at_100
      value: 34.86408333333333
    - type: map_at_1000
      value: 34.9795
    - type: map_at_3
      value: 31.259999999999998
    - type: map_at_5
      value: 32.71541666666666
    - type: mrr_at_1
      value: 30.863749999999996
    - type: mrr_at_10
      value: 37.99183333333333
    - type: mrr_at_100
      value: 38.790499999999994
    - type: mrr_at_1000
      value: 38.85575000000001
    - type: mrr_at_3
      value: 35.82083333333333
    - type: mrr_at_5
      value: 37.07533333333333
    - type: ndcg_at_1
      value: 30.863749999999996
    - type: ndcg_at_10
      value: 38.52141666666667
    - type: ndcg_at_100
      value: 43.17966666666667
    - type: ndcg_at_1000
      value: 45.64608333333333
    - type: ndcg_at_3
      value: 34.333000000000006
    - type: ndcg_at_5
      value: 36.34975
    - type: precision_at_1
      value: 30.863749999999996
    - type: precision_at_10
      value: 6.598999999999999
    - type: precision_at_100
      value: 1.0502500000000001
    - type: precision_at_1000
      value: 0.14400000000000002
    - type: precision_at_3
      value: 15.557583333333334
    - type: precision_at_5
      value: 11.020000000000001
    - type: recall_at_1
      value: 25.946583333333333
    - type: recall_at_10
      value: 48.36991666666666
    - type: recall_at_100
      value: 69.02408333333334
    - type: recall_at_1000
      value: 86.43858333333331
    - type: recall_at_3
      value: 36.4965
    - type: recall_at_5
      value: 41.76258333333334
    - type: map_at_1
      value: 22.431
    - type: map_at_10
      value: 28.889
    - type: map_at_100
      value: 29.642000000000003
    - type: map_at_1000
      value: 29.742
    - type: map_at_3
      value: 26.998
    - type: map_at_5
      value: 28.172000000000004
    - type: mrr_at_1
      value: 25.307000000000002
    - type: mrr_at_10
      value: 31.763
    - type: mrr_at_100
      value: 32.443
    - type: mrr_at_1000
      value: 32.531
    - type: mrr_at_3
      value: 29.959000000000003
    - type: mrr_at_5
      value: 31.063000000000002
    - type: ndcg_at_1
      value: 25.307000000000002
    - type: ndcg_at_10
      value: 32.586999999999996
    - type: ndcg_at_100
      value: 36.5
    - type: ndcg_at_1000
      value: 39.133
    - type: ndcg_at_3
      value: 29.25
    - type: ndcg_at_5
      value: 31.023
    - type: precision_at_1
      value: 25.307000000000002
    - type: precision_at_10
      value: 4.954
    - type: precision_at_100
      value: 0.747
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 12.577
    - type: precision_at_5
      value: 8.741999999999999
    - type: recall_at_1
      value: 22.431
    - type: recall_at_10
      value: 41.134
    - type: recall_at_100
      value: 59.28600000000001
    - type: recall_at_1000
      value: 78.857
    - type: recall_at_3
      value: 31.926
    - type: recall_at_5
      value: 36.335
    - type: map_at_1
      value: 17.586
    - type: map_at_10
      value: 23.304
    - type: map_at_100
      value: 24.159
    - type: map_at_1000
      value: 24.281
    - type: map_at_3
      value: 21.316
    - type: map_at_5
      value: 22.383
    - type: mrr_at_1
      value: 21.645
    - type: mrr_at_10
      value: 27.365000000000002
    - type: mrr_at_100
      value: 28.108
    - type: mrr_at_1000
      value: 28.192
    - type: mrr_at_3
      value: 25.482
    - type: mrr_at_5
      value: 26.479999999999997
    - type: ndcg_at_1
      value: 21.645
    - type: ndcg_at_10
      value: 27.306
    - type: ndcg_at_100
      value: 31.496000000000002
    - type: ndcg_at_1000
      value: 34.53
    - type: ndcg_at_3
      value: 23.73
    - type: ndcg_at_5
      value: 25.294
    - type: precision_at_1
      value: 21.645
    - type: precision_at_10
      value: 4.797
    - type: precision_at_100
      value: 0.8059999999999999
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 10.850999999999999
    - type: precision_at_5
      value: 7.736
    - type: recall_at_1
      value: 17.586
    - type: recall_at_10
      value: 35.481
    - type: recall_at_100
      value: 54.534000000000006
    - type: recall_at_1000
      value: 76.456
    - type: recall_at_3
      value: 25.335
    - type: recall_at_5
      value: 29.473
    - type: map_at_1
      value: 25.095
    - type: map_at_10
      value: 32.374
    - type: map_at_100
      value: 33.537
    - type: map_at_1000
      value: 33.634
    - type: map_at_3
      value: 30.089
    - type: map_at_5
      value: 31.433
    - type: mrr_at_1
      value: 29.198
    - type: mrr_at_10
      value: 36.01
    - type: mrr_at_100
      value: 37.022
    - type: mrr_at_1000
      value: 37.083
    - type: mrr_at_3
      value: 33.94
    - type: mrr_at_5
      value: 35.148
    - type: ndcg_at_1
      value: 29.198
    - type: ndcg_at_10
      value: 36.729
    - type: ndcg_at_100
      value: 42.114000000000004
    - type: ndcg_at_1000
      value: 44.592
    - type: ndcg_at_3
      value: 32.644
    - type: ndcg_at_5
      value: 34.652
    - type: precision_at_1
      value: 29.198
    - type: precision_at_10
      value: 5.970000000000001
    - type: precision_at_100
      value: 0.967
    - type: precision_at_1000
      value: 0.129
    - type: precision_at_3
      value: 14.396999999999998
    - type: precision_at_5
      value: 10.093
    - type: recall_at_1
      value: 25.095
    - type: recall_at_10
      value: 46.392
    - type: recall_at_100
      value: 69.706
    - type: recall_at_1000
      value: 87.738
    - type: recall_at_3
      value: 35.303000000000004
    - type: recall_at_5
      value: 40.441
    - type: map_at_1
      value: 26.857999999999997
    - type: map_at_10
      value: 34.066
    - type: map_at_100
      value: 35.671
    - type: map_at_1000
      value: 35.881
    - type: map_at_3
      value: 31.304
    - type: map_at_5
      value: 32.885
    - type: mrr_at_1
      value: 32.411
    - type: mrr_at_10
      value: 38.987
    - type: mrr_at_100
      value: 39.894
    - type: mrr_at_1000
      value: 39.959
    - type: mrr_at_3
      value: 36.626999999999995
    - type: mrr_at_5
      value: 38.011
    - type: ndcg_at_1
      value: 32.411
    - type: ndcg_at_10
      value: 39.208
    - type: ndcg_at_100
      value: 44.626
    - type: ndcg_at_1000
      value: 47.43
    - type: ndcg_at_3
      value: 35.091
    - type: ndcg_at_5
      value: 37.119
    - type: precision_at_1
      value: 32.411
    - type: precision_at_10
      value: 7.51
    - type: precision_at_100
      value: 1.486
    - type: precision_at_1000
      value: 0.234
    - type: precision_at_3
      value: 16.14
    - type: precision_at_5
      value: 11.976
    - type: recall_at_1
      value: 26.857999999999997
    - type: recall_at_10
      value: 47.407
    - type: recall_at_100
      value: 72.236
    - type: recall_at_1000
      value: 90.77
    - type: recall_at_3
      value: 35.125
    - type: recall_at_5
      value: 40.522999999999996
    - type: map_at_1
      value: 21.3
    - type: map_at_10
      value: 27.412999999999997
    - type: map_at_100
      value: 28.29
    - type: map_at_1000
      value: 28.398
    - type: map_at_3
      value: 25.169999999999998
    - type: map_at_5
      value: 26.496
    - type: mrr_at_1
      value: 23.29
    - type: mrr_at_10
      value: 29.215000000000003
    - type: mrr_at_100
      value: 30.073
    - type: mrr_at_1000
      value: 30.156
    - type: mrr_at_3
      value: 26.956000000000003
    - type: mrr_at_5
      value: 28.38
    - type: ndcg_at_1
      value: 23.29
    - type: ndcg_at_10
      value: 31.113000000000003
    - type: ndcg_at_100
      value: 35.701
    - type: ndcg_at_1000
      value: 38.505
    - type: ndcg_at_3
      value: 26.727
    - type: ndcg_at_5
      value: 29.037000000000003
    - type: precision_at_1
      value: 23.29
    - type: precision_at_10
      value: 4.787
    - type: precision_at_100
      value: 0.763
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 11.091
    - type: precision_at_5
      value: 7.985
    - type: recall_at_1
      value: 21.3
    - type: recall_at_10
      value: 40.782000000000004
    - type: recall_at_100
      value: 62.13999999999999
    - type: recall_at_1000
      value: 83.012
    - type: recall_at_3
      value: 29.131
    - type: recall_at_5
      value: 34.624
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
      value: 9.631
    - type: map_at_10
      value: 16.634999999999998
    - type: map_at_100
      value: 18.23
    - type: map_at_1000
      value: 18.419
    - type: map_at_3
      value: 13.66
    - type: map_at_5
      value: 15.173
    - type: mrr_at_1
      value: 21.368000000000002
    - type: mrr_at_10
      value: 31.56
    - type: mrr_at_100
      value: 32.58
    - type: mrr_at_1000
      value: 32.633
    - type: mrr_at_3
      value: 28.241
    - type: mrr_at_5
      value: 30.225
    - type: ndcg_at_1
      value: 21.368000000000002
    - type: ndcg_at_10
      value: 23.855999999999998
    - type: ndcg_at_100
      value: 30.686999999999998
    - type: ndcg_at_1000
      value: 34.327000000000005
    - type: ndcg_at_3
      value: 18.781
    - type: ndcg_at_5
      value: 20.73
    - type: precision_at_1
      value: 21.368000000000002
    - type: precision_at_10
      value: 7.564
    - type: precision_at_100
      value: 1.496
    - type: precision_at_1000
      value: 0.217
    - type: precision_at_3
      value: 13.876
    - type: precision_at_5
      value: 11.062
    - type: recall_at_1
      value: 9.631
    - type: recall_at_10
      value: 29.517
    - type: recall_at_100
      value: 53.452
    - type: recall_at_1000
      value: 74.115
    - type: recall_at_3
      value: 17.605999999999998
    - type: recall_at_5
      value: 22.505
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
      value: 8.885
    - type: map_at_10
      value: 18.798000000000002
    - type: map_at_100
      value: 26.316
    - type: map_at_1000
      value: 27.869
    - type: map_at_3
      value: 13.719000000000001
    - type: map_at_5
      value: 15.716
    - type: mrr_at_1
      value: 66
    - type: mrr_at_10
      value: 74.263
    - type: mrr_at_100
      value: 74.519
    - type: mrr_at_1000
      value: 74.531
    - type: mrr_at_3
      value: 72.458
    - type: mrr_at_5
      value: 73.321
    - type: ndcg_at_1
      value: 53.87499999999999
    - type: ndcg_at_10
      value: 40.355999999999995
    - type: ndcg_at_100
      value: 44.366
    - type: ndcg_at_1000
      value: 51.771
    - type: ndcg_at_3
      value: 45.195
    - type: ndcg_at_5
      value: 42.187000000000005
    - type: precision_at_1
      value: 66
    - type: precision_at_10
      value: 31.75
    - type: precision_at_100
      value: 10.11
    - type: precision_at_1000
      value: 1.9800000000000002
    - type: precision_at_3
      value: 48.167
    - type: precision_at_5
      value: 40.050000000000004
    - type: recall_at_1
      value: 8.885
    - type: recall_at_10
      value: 24.471999999999998
    - type: recall_at_100
      value: 49.669000000000004
    - type: recall_at_1000
      value: 73.383
    - type: recall_at_3
      value: 14.872
    - type: recall_at_5
      value: 18.262999999999998
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
      value: 45.18
    - type: f1
      value: 40.26878691789978
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
      value: 62.751999999999995
    - type: map_at_10
      value: 74.131
    - type: map_at_100
      value: 74.407
    - type: map_at_1000
      value: 74.423
    - type: map_at_3
      value: 72.329
    - type: map_at_5
      value: 73.555
    - type: mrr_at_1
      value: 67.282
    - type: mrr_at_10
      value: 78.292
    - type: mrr_at_100
      value: 78.455
    - type: mrr_at_1000
      value: 78.458
    - type: mrr_at_3
      value: 76.755
    - type: mrr_at_5
      value: 77.839
    - type: ndcg_at_1
      value: 67.282
    - type: ndcg_at_10
      value: 79.443
    - type: ndcg_at_100
      value: 80.529
    - type: ndcg_at_1000
      value: 80.812
    - type: ndcg_at_3
      value: 76.281
    - type: ndcg_at_5
      value: 78.235
    - type: precision_at_1
      value: 67.282
    - type: precision_at_10
      value: 10.078
    - type: precision_at_100
      value: 1.082
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 30.178
    - type: precision_at_5
      value: 19.232
    - type: recall_at_1
      value: 62.751999999999995
    - type: recall_at_10
      value: 91.521
    - type: recall_at_100
      value: 95.997
    - type: recall_at_1000
      value: 97.775
    - type: recall_at_3
      value: 83.131
    - type: recall_at_5
      value: 87.93299999999999
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
      value: 18.861
    - type: map_at_10
      value: 30.252000000000002
    - type: map_at_100
      value: 32.082
    - type: map_at_1000
      value: 32.261
    - type: map_at_3
      value: 25.909
    - type: map_at_5
      value: 28.296
    - type: mrr_at_1
      value: 37.346000000000004
    - type: mrr_at_10
      value: 45.802
    - type: mrr_at_100
      value: 46.611999999999995
    - type: mrr_at_1000
      value: 46.659
    - type: mrr_at_3
      value: 43.056
    - type: mrr_at_5
      value: 44.637
    - type: ndcg_at_1
      value: 37.346000000000004
    - type: ndcg_at_10
      value: 38.169
    - type: ndcg_at_100
      value: 44.864
    - type: ndcg_at_1000
      value: 47.974
    - type: ndcg_at_3
      value: 33.619
    - type: ndcg_at_5
      value: 35.317
    - type: precision_at_1
      value: 37.346000000000004
    - type: precision_at_10
      value: 10.693999999999999
    - type: precision_at_100
      value: 1.775
    - type: precision_at_1000
      value: 0.231
    - type: precision_at_3
      value: 22.325
    - type: precision_at_5
      value: 16.852
    - type: recall_at_1
      value: 18.861
    - type: recall_at_10
      value: 45.672000000000004
    - type: recall_at_100
      value: 70.60499999999999
    - type: recall_at_1000
      value: 89.216
    - type: recall_at_3
      value: 30.361
    - type: recall_at_5
      value: 36.998999999999995
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
      value: 37.852999999999994
    - type: map_at_10
      value: 59.961
    - type: map_at_100
      value: 60.78
    - type: map_at_1000
      value: 60.843
    - type: map_at_3
      value: 56.39999999999999
    - type: map_at_5
      value: 58.646
    - type: mrr_at_1
      value: 75.70599999999999
    - type: mrr_at_10
      value: 82.321
    - type: mrr_at_100
      value: 82.516
    - type: mrr_at_1000
      value: 82.525
    - type: mrr_at_3
      value: 81.317
    - type: mrr_at_5
      value: 81.922
    - type: ndcg_at_1
      value: 75.70599999999999
    - type: ndcg_at_10
      value: 68.557
    - type: ndcg_at_100
      value: 71.485
    - type: ndcg_at_1000
      value: 72.71600000000001
    - type: ndcg_at_3
      value: 63.524
    - type: ndcg_at_5
      value: 66.338
    - type: precision_at_1
      value: 75.70599999999999
    - type: precision_at_10
      value: 14.463000000000001
    - type: precision_at_100
      value: 1.677
    - type: precision_at_1000
      value: 0.184
    - type: precision_at_3
      value: 40.806
    - type: precision_at_5
      value: 26.709
    - type: recall_at_1
      value: 37.852999999999994
    - type: recall_at_10
      value: 72.316
    - type: recall_at_100
      value: 83.842
    - type: recall_at_1000
      value: 91.999
    - type: recall_at_3
      value: 61.209
    - type: recall_at_5
      value: 66.77199999999999
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
      value: 85.46039999999999
    - type: ap
      value: 79.9812521351881
    - type: f1
      value: 85.31722909702084
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
      value: 22.704
    - type: map_at_10
      value: 35.329
    - type: map_at_100
      value: 36.494
    - type: map_at_1000
      value: 36.541000000000004
    - type: map_at_3
      value: 31.476
    - type: map_at_5
      value: 33.731
    - type: mrr_at_1
      value: 23.294999999999998
    - type: mrr_at_10
      value: 35.859
    - type: mrr_at_100
      value: 36.968
    - type: mrr_at_1000
      value: 37.008
    - type: mrr_at_3
      value: 32.085
    - type: mrr_at_5
      value: 34.299
    - type: ndcg_at_1
      value: 23.324
    - type: ndcg_at_10
      value: 42.274
    - type: ndcg_at_100
      value: 47.839999999999996
    - type: ndcg_at_1000
      value: 48.971
    - type: ndcg_at_3
      value: 34.454
    - type: ndcg_at_5
      value: 38.464
    - type: precision_at_1
      value: 23.324
    - type: precision_at_10
      value: 6.648
    - type: precision_at_100
      value: 0.9440000000000001
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.674999999999999
    - type: precision_at_5
      value: 10.850999999999999
    - type: recall_at_1
      value: 22.704
    - type: recall_at_10
      value: 63.660000000000004
    - type: recall_at_100
      value: 89.29899999999999
    - type: recall_at_1000
      value: 97.88900000000001
    - type: recall_at_3
      value: 42.441
    - type: recall_at_5
      value: 52.04
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
      value: 93.1326949384405
    - type: f1
      value: 92.89743579612082
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (de)
      type: mteb/mtop_domain
      config: de
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 89.62524654832347
    - type: f1
      value: 88.65106082263151
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (es)
      type: mteb/mtop_domain
      config: es
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 90.59039359573046
    - type: f1
      value: 90.31532892105662
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (fr)
      type: mteb/mtop_domain
      config: fr
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 86.21046038208581
    - type: f1
      value: 86.41459529813113
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (hi)
      type: mteb/mtop_domain
      config: hi
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 87.3180351380423
    - type: f1
      value: 86.71383078226444
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPDomainClassification (th)
      type: mteb/mtop_domain
      config: th
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 86.24231464737792
    - type: f1
      value: 86.31845567592403
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
      value: 75.27131782945736
    - type: f1
      value: 57.52079940417103
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (de)
      type: mteb/mtop_intent
      config: de
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 71.2341504649197
    - type: f1
      value: 51.349951558039244
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (es)
      type: mteb/mtop_intent
      config: es
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 71.27418278852569
    - type: f1
      value: 50.1714985749095
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (fr)
      type: mteb/mtop_intent
      config: fr
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 67.68243031631694
    - type: f1
      value: 50.1066160836192
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (hi)
      type: mteb/mtop_intent
      config: hi
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 69.2362854069559
    - type: f1
      value: 48.821279948766424
  - task:
      type: Classification
    dataset:
      name: MTEB MTOPIntentClassification (th)
      type: mteb/mtop_intent
      config: th
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 71.71428571428571
    - type: f1
      value: 53.94611389496195
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (af)
      type: mteb/amazon_massive_intent
      config: af
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.97646267652992
    - type: f1
      value: 57.26797883561521
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (am)
      type: mteb/amazon_massive_intent
      config: am
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 53.65501008742435
    - type: f1
      value: 50.416258382177034
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ar)
      type: mteb/amazon_massive_intent
      config: ar
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 57.45796906523201
    - type: f1
      value: 53.306690547422185
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (az)
      type: mteb/amazon_massive_intent
      config: az
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 62.59246805648957
    - type: f1
      value: 59.818381969051494
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (bn)
      type: mteb/amazon_massive_intent
      config: bn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 61.126429051782104
    - type: f1
      value: 58.25993593933026
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (cy)
      type: mteb/amazon_massive_intent
      config: cy
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 50.057162071284466
    - type: f1
      value: 46.96095728790911
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (da)
      type: mteb/amazon_massive_intent
      config: da
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.64425016812375
    - type: f1
      value: 62.858291698755764
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (de)
      type: mteb/amazon_massive_intent
      config: de
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.08944182918628
    - type: f1
      value: 62.44639030604241
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (el)
      type: mteb/amazon_massive_intent
      config: el
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 64.68056489576328
    - type: f1
      value: 61.775326758789504
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
      value: 72.11163416274377
    - type: f1
      value: 69.70789096927015
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (es)
      type: mteb/amazon_massive_intent
      config: es
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.40282447881641
    - type: f1
      value: 66.38492065671895
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (fa)
      type: mteb/amazon_massive_intent
      config: fa
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.24613315400134
    - type: f1
      value: 64.3348019501336
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (fi)
      type: mteb/amazon_massive_intent
      config: fi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 65.78345662407531
    - type: f1
      value: 62.21279452354622
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (fr)
      type: mteb/amazon_massive_intent
      config: fr
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.9455279085407
    - type: f1
      value: 65.48193124964094
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (he)
      type: mteb/amazon_massive_intent
      config: he
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 62.05110961667788
    - type: f1
      value: 58.097856564684534
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (hi)
      type: mteb/amazon_massive_intent
      config: hi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 64.95292535305985
    - type: f1
      value: 62.09182174767901
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (hu)
      type: mteb/amazon_massive_intent
      config: hu
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 64.97310020174848
    - type: f1
      value: 61.14252567730396
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (hy)
      type: mteb/amazon_massive_intent
      config: hy
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 60.08069939475453
    - type: f1
      value: 57.044041742492034
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (id)
      type: mteb/amazon_massive_intent
      config: id
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.63752521856085
    - type: f1
      value: 63.889340907205316
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (is)
      type: mteb/amazon_massive_intent
      config: is
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 56.385339609952936
    - type: f1
      value: 53.449033750088304
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (it)
      type: mteb/amazon_massive_intent
      config: it
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.93073301950234
    - type: f1
      value: 65.9884357824104
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ja)
      type: mteb/amazon_massive_intent
      config: ja
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.94418291862812
    - type: f1
      value: 66.48740222583132
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (jv)
      type: mteb/amazon_massive_intent
      config: jv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.26025554808339
    - type: f1
      value: 50.19562815100793
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ka)
      type: mteb/amazon_massive_intent
      config: ka
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 48.98789509078682
    - type: f1
      value: 46.65788438676836
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (km)
      type: mteb/amazon_massive_intent
      config: km
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 44.68728984532616
    - type: f1
      value: 41.642419349541996
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (kn)
      type: mteb/amazon_massive_intent
      config: kn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.19300605245461
    - type: f1
      value: 55.8626492442437
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ko)
      type: mteb/amazon_massive_intent
      config: ko
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.33826496301278
    - type: f1
      value: 63.89499791648792
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (lv)
      type: mteb/amazon_massive_intent
      config: lv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 60.33960995292536
    - type: f1
      value: 57.15242464180892
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ml)
      type: mteb/amazon_massive_intent
      config: ml
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 63.09347679892402
    - type: f1
      value: 59.64733214063841
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (mn)
      type: mteb/amazon_massive_intent
      config: mn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.75924680564896
    - type: f1
      value: 55.96585692366827
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ms)
      type: mteb/amazon_massive_intent
      config: ms
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 62.48486886348352
    - type: f1
      value: 59.45143559032946
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (my)
      type: mteb/amazon_massive_intent
      config: my
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.56422326832549
    - type: f1
      value: 54.96368702901926
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (nb)
      type: mteb/amazon_massive_intent
      config: nb
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.18022864828512
    - type: f1
      value: 63.05369805040634
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (nl)
      type: mteb/amazon_massive_intent
      config: nl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.30329522528581
    - type: f1
      value: 64.06084612020727
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (pl)
      type: mteb/amazon_massive_intent
      config: pl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.36919973100201
    - type: f1
      value: 65.12154124788887
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (pt)
      type: mteb/amazon_massive_intent
      config: pt
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.98117014122394
    - type: f1
      value: 66.41847559806962
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ro)
      type: mteb/amazon_massive_intent
      config: ro
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 65.53799596503026
    - type: f1
      value: 62.17067330740817
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ru)
      type: mteb/amazon_massive_intent
      config: ru
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.01815736381977
    - type: f1
      value: 66.24988369607843
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sl)
      type: mteb/amazon_massive_intent
      config: sl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 62.34700739744452
    - type: f1
      value: 59.957933424941636
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sq)
      type: mteb/amazon_massive_intent
      config: sq
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 61.23402824478815
    - type: f1
      value: 57.98836976018471
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sv)
      type: mteb/amazon_massive_intent
      config: sv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.54068594485541
    - type: f1
      value: 65.43849680666855
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (sw)
      type: mteb/amazon_massive_intent
      config: sw
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 55.998655010087425
    - type: f1
      value: 52.83737515406804
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ta)
      type: mteb/amazon_massive_intent
      config: ta
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.71217215870882
    - type: f1
      value: 55.051794977833026
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (te)
      type: mteb/amazon_massive_intent
      config: te
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 59.724277067921996
    - type: f1
      value: 56.33485571838306
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (th)
      type: mteb/amazon_massive_intent
      config: th
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 65.59515803631473
    - type: f1
      value: 64.96772366193588
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (tl)
      type: mteb/amazon_massive_intent
      config: tl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 60.860793544048406
    - type: f1
      value: 58.148845819115394
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (tr)
      type: mteb/amazon_massive_intent
      config: tr
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.40753194351043
    - type: f1
      value: 63.18903778054698
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (ur)
      type: mteb/amazon_massive_intent
      config: ur
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 61.52320107599194
    - type: f1
      value: 58.356144563398516
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (vi)
      type: mteb/amazon_massive_intent
      config: vi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.17014122394083
    - type: f1
      value: 63.919964062638925
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
      value: 69.15601882985878
    - type: f1
      value: 67.01451905761371
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveIntentClassification (zh-TW)
      type: mteb/amazon_massive_intent
      config: zh-TW
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 64.65030262273034
    - type: f1
      value: 64.14420425129063
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (af)
      type: mteb/amazon_massive_scenario
      config: af
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 65.08742434431743
    - type: f1
      value: 63.044060042311756
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (am)
      type: mteb/amazon_massive_scenario
      config: am
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 58.52387357094821
    - type: f1
      value: 56.82398588814534
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ar)
      type: mteb/amazon_massive_scenario
      config: ar
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.239408204438476
    - type: f1
      value: 61.92570286170469
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (az)
      type: mteb/amazon_massive_scenario
      config: az
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.74915938130463
    - type: f1
      value: 62.130740689396276
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (bn)
      type: mteb/amazon_massive_scenario
      config: bn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 65.00336247478144
    - type: f1
      value: 63.71080635228055
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (cy)
      type: mteb/amazon_massive_scenario
      config: cy
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 52.837928715534645
    - type: f1
      value: 50.390741680320836
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (da)
      type: mteb/amazon_massive_scenario
      config: da
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.42098184263618
    - type: f1
      value: 71.41355113538995
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (de)
      type: mteb/amazon_massive_scenario
      config: de
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.95359784801613
    - type: f1
      value: 71.42699340156742
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (el)
      type: mteb/amazon_massive_scenario
      config: el
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.18157363819772
    - type: f1
      value: 69.74836113037671
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
      value: 77.08137188971082
    - type: f1
      value: 76.78000685068261
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (es)
      type: mteb/amazon_massive_scenario
      config: es
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.5030262273033
    - type: f1
      value: 71.71620130425673
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (fa)
      type: mteb/amazon_massive_scenario
      config: fa
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.24546065904505
    - type: f1
      value: 69.07638311730359
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (fi)
      type: mteb/amazon_massive_scenario
      config: fi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 69.12911903160726
    - type: f1
      value: 68.32651736539815
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (fr)
      type: mteb/amazon_massive_scenario
      config: fr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.89307330195025
    - type: f1
      value: 71.33986549860187
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (he)
      type: mteb/amazon_massive_scenario
      config: he
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 67.44451916610626
    - type: f1
      value: 66.90192664503866
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (hi)
      type: mteb/amazon_massive_scenario
      config: hi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 69.16274377942166
    - type: f1
      value: 68.01090953775066
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (hu)
      type: mteb/amazon_massive_scenario
      config: hu
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.75319435104237
    - type: f1
      value: 70.18035309201403
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (hy)
      type: mteb/amazon_massive_scenario
      config: hy
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 63.14391392064559
    - type: f1
      value: 61.48286540778145
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (id)
      type: mteb/amazon_massive_scenario
      config: id
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.70275722932078
    - type: f1
      value: 70.26164779846495
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (is)
      type: mteb/amazon_massive_scenario
      config: is
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 60.93813046402153
    - type: f1
      value: 58.8852862116525
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (it)
      type: mteb/amazon_massive_scenario
      config: it
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.320107599193
    - type: f1
      value: 72.19836409602924
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ja)
      type: mteb/amazon_massive_scenario
      config: ja
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.65366509751176
    - type: f1
      value: 74.55188288799579
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (jv)
      type: mteb/amazon_massive_scenario
      config: jv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 59.694014794889036
    - type: f1
      value: 58.11353311721067
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ka)
      type: mteb/amazon_massive_scenario
      config: ka
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 54.37457969065231
    - type: f1
      value: 52.81306134311697
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (km)
      type: mteb/amazon_massive_scenario
      config: km
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 48.3086751849361
    - type: f1
      value: 45.396449765419376
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (kn)
      type: mteb/amazon_massive_scenario
      config: kn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.151983860121064
    - type: f1
      value: 60.31762544281696
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ko)
      type: mteb/amazon_massive_scenario
      config: ko
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.44788164088769
    - type: f1
      value: 71.68150151736367
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (lv)
      type: mteb/amazon_massive_scenario
      config: lv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.81439139206455
    - type: f1
      value: 62.06735559105593
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ml)
      type: mteb/amazon_massive_scenario
      config: ml
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 68.04303967720242
    - type: f1
      value: 66.68298851670133
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (mn)
      type: mteb/amazon_massive_scenario
      config: mn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 61.43913920645595
    - type: f1
      value: 60.25605977560783
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ms)
      type: mteb/amazon_massive_scenario
      config: ms
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 66.90316072629456
    - type: f1
      value: 65.1325924692381
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (my)
      type: mteb/amazon_massive_scenario
      config: my
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 61.63752521856086
    - type: f1
      value: 59.14284778039585
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (nb)
      type: mteb/amazon_massive_scenario
      config: nb
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.63080026899797
    - type: f1
      value: 70.89771864626877
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (nl)
      type: mteb/amazon_massive_scenario
      config: nl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.10827168796234
    - type: f1
      value: 71.71954219691159
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (pl)
      type: mteb/amazon_massive_scenario
      config: pl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.59515803631471
    - type: f1
      value: 70.05040128099003
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (pt)
      type: mteb/amazon_massive_scenario
      config: pt
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.83389374579691
    - type: f1
      value: 70.84877936562735
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ro)
      type: mteb/amazon_massive_scenario
      config: ro
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 69.18628110289173
    - type: f1
      value: 68.97232927921841
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ru)
      type: mteb/amazon_massive_scenario
      config: ru
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.99260255548083
    - type: f1
      value: 72.85139492157732
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sl)
      type: mteb/amazon_massive_scenario
      config: sl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 65.26227303295225
    - type: f1
      value: 65.08833655469431
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sq)
      type: mteb/amazon_massive_scenario
      config: sq
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 66.48621385339611
    - type: f1
      value: 64.43483199071298
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sv)
      type: mteb/amazon_massive_scenario
      config: sv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 73.14391392064559
    - type: f1
      value: 72.2580822579741
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (sw)
      type: mteb/amazon_massive_scenario
      config: sw
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 59.88567585743107
    - type: f1
      value: 58.3073765932569
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ta)
      type: mteb/amazon_massive_scenario
      config: ta
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.38399462004034
    - type: f1
      value: 60.82139544252606
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (te)
      type: mteb/amazon_massive_scenario
      config: te
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.58574310692671
    - type: f1
      value: 60.71443370385374
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (th)
      type: mteb/amazon_massive_scenario
      config: th
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.61398789509079
    - type: f1
      value: 70.99761812049401
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (tl)
      type: mteb/amazon_massive_scenario
      config: tl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 62.73705447209146
    - type: f1
      value: 61.680849331794796
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (tr)
      type: mteb/amazon_massive_scenario
      config: tr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.66778749159381
    - type: f1
      value: 71.17320646080115
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (ur)
      type: mteb/amazon_massive_scenario
      config: ur
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 64.640215198386
    - type: f1
      value: 63.301805157015444
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (vi)
      type: mteb/amazon_massive_scenario
      config: vi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.00672494956288
    - type: f1
      value: 70.26005548582106
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
      value: 75.42030934767989
    - type: f1
      value: 75.2074842882598
  - task:
      type: Classification
    dataset:
      name: MTEB MassiveScenarioClassification (zh-TW)
      type: mteb/amazon_massive_scenario
      config: zh-TW
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.69266980497646
    - type: f1
      value: 70.94103167391192
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
      value: 28.91697191169135
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
      value: 28.434000079573313
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
      value: 30.96683513343383
    - type: mrr
      value: 31.967364078714834
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
      value: 5.5280000000000005
    - type: map_at_10
      value: 11.793
    - type: map_at_100
      value: 14.496999999999998
    - type: map_at_1000
      value: 15.783
    - type: map_at_3
      value: 8.838
    - type: map_at_5
      value: 10.07
    - type: mrr_at_1
      value: 43.653
    - type: mrr_at_10
      value: 51.531000000000006
    - type: mrr_at_100
      value: 52.205
    - type: mrr_at_1000
      value: 52.242999999999995
    - type: mrr_at_3
      value: 49.431999999999995
    - type: mrr_at_5
      value: 50.470000000000006
    - type: ndcg_at_1
      value: 42.415000000000006
    - type: ndcg_at_10
      value: 32.464999999999996
    - type: ndcg_at_100
      value: 28.927999999999997
    - type: ndcg_at_1000
      value: 37.629000000000005
    - type: ndcg_at_3
      value: 37.845
    - type: ndcg_at_5
      value: 35.147
    - type: precision_at_1
      value: 43.653
    - type: precision_at_10
      value: 23.932000000000002
    - type: precision_at_100
      value: 7.17
    - type: precision_at_1000
      value: 1.967
    - type: precision_at_3
      value: 35.397
    - type: precision_at_5
      value: 29.907
    - type: recall_at_1
      value: 5.5280000000000005
    - type: recall_at_10
      value: 15.568000000000001
    - type: recall_at_100
      value: 28.54
    - type: recall_at_1000
      value: 59.864
    - type: recall_at_3
      value: 9.822000000000001
    - type: recall_at_5
      value: 11.726
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
      value: 37.041000000000004
    - type: map_at_10
      value: 52.664
    - type: map_at_100
      value: 53.477
    - type: map_at_1000
      value: 53.505
    - type: map_at_3
      value: 48.510999999999996
    - type: map_at_5
      value: 51.036
    - type: mrr_at_1
      value: 41.338
    - type: mrr_at_10
      value: 55.071000000000005
    - type: mrr_at_100
      value: 55.672
    - type: mrr_at_1000
      value: 55.689
    - type: mrr_at_3
      value: 51.82
    - type: mrr_at_5
      value: 53.852
    - type: ndcg_at_1
      value: 41.338
    - type: ndcg_at_10
      value: 60.01800000000001
    - type: ndcg_at_100
      value: 63.409000000000006
    - type: ndcg_at_1000
      value: 64.017
    - type: ndcg_at_3
      value: 52.44799999999999
    - type: ndcg_at_5
      value: 56.571000000000005
    - type: precision_at_1
      value: 41.338
    - type: precision_at_10
      value: 9.531
    - type: precision_at_100
      value: 1.145
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 23.416
    - type: precision_at_5
      value: 16.46
    - type: recall_at_1
      value: 37.041000000000004
    - type: recall_at_10
      value: 79.76299999999999
    - type: recall_at_100
      value: 94.39
    - type: recall_at_1000
      value: 98.851
    - type: recall_at_3
      value: 60.465
    - type: recall_at_5
      value: 69.906
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
      value: 69.952
    - type: map_at_10
      value: 83.758
    - type: map_at_100
      value: 84.406
    - type: map_at_1000
      value: 84.425
    - type: map_at_3
      value: 80.839
    - type: map_at_5
      value: 82.646
    - type: mrr_at_1
      value: 80.62
    - type: mrr_at_10
      value: 86.947
    - type: mrr_at_100
      value: 87.063
    - type: mrr_at_1000
      value: 87.064
    - type: mrr_at_3
      value: 85.96000000000001
    - type: mrr_at_5
      value: 86.619
    - type: ndcg_at_1
      value: 80.63
    - type: ndcg_at_10
      value: 87.64800000000001
    - type: ndcg_at_100
      value: 88.929
    - type: ndcg_at_1000
      value: 89.054
    - type: ndcg_at_3
      value: 84.765
    - type: ndcg_at_5
      value: 86.291
    - type: precision_at_1
      value: 80.63
    - type: precision_at_10
      value: 13.314
    - type: precision_at_100
      value: 1.525
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.1
    - type: precision_at_5
      value: 24.372
    - type: recall_at_1
      value: 69.952
    - type: recall_at_10
      value: 94.955
    - type: recall_at_100
      value: 99.38
    - type: recall_at_1000
      value: 99.96000000000001
    - type: recall_at_3
      value: 86.60600000000001
    - type: recall_at_5
      value: 90.997
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
      value: 42.41329517878427
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
      value: 55.171278362748666
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
      value: 4.213
    - type: map_at_10
      value: 9.895
    - type: map_at_100
      value: 11.776
    - type: map_at_1000
      value: 12.084
    - type: map_at_3
      value: 7.2669999999999995
    - type: map_at_5
      value: 8.620999999999999
    - type: mrr_at_1
      value: 20.8
    - type: mrr_at_10
      value: 31.112000000000002
    - type: mrr_at_100
      value: 32.274
    - type: mrr_at_1000
      value: 32.35
    - type: mrr_at_3
      value: 28.133000000000003
    - type: mrr_at_5
      value: 29.892999999999997
    - type: ndcg_at_1
      value: 20.8
    - type: ndcg_at_10
      value: 17.163999999999998
    - type: ndcg_at_100
      value: 24.738
    - type: ndcg_at_1000
      value: 30.316
    - type: ndcg_at_3
      value: 16.665
    - type: ndcg_at_5
      value: 14.478
    - type: precision_at_1
      value: 20.8
    - type: precision_at_10
      value: 8.74
    - type: precision_at_100
      value: 1.963
    - type: precision_at_1000
      value: 0.33
    - type: precision_at_3
      value: 15.467
    - type: precision_at_5
      value: 12.6
    - type: recall_at_1
      value: 4.213
    - type: recall_at_10
      value: 17.698
    - type: recall_at_100
      value: 39.838
    - type: recall_at_1000
      value: 66.893
    - type: recall_at_3
      value: 9.418
    - type: recall_at_5
      value: 12.773000000000001
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
      value: 82.90453315738294
    - type: cos_sim_spearman
      value: 78.51197850080254
    - type: euclidean_pearson
      value: 80.09647123597748
    - type: euclidean_spearman
      value: 78.63548011514061
    - type: manhattan_pearson
      value: 80.10645285675231
    - type: manhattan_spearman
      value: 78.57861806068901
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
      value: 84.2616156846401
    - type: cos_sim_spearman
      value: 76.69713867850156
    - type: euclidean_pearson
      value: 77.97948563800394
    - type: euclidean_spearman
      value: 74.2371211567807
    - type: manhattan_pearson
      value: 77.69697879669705
    - type: manhattan_spearman
      value: 73.86529778022278
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
      value: 77.0293269315045
    - type: cos_sim_spearman
      value: 78.02555120584198
    - type: euclidean_pearson
      value: 78.25398100379078
    - type: euclidean_spearman
      value: 78.66963870599464
    - type: manhattan_pearson
      value: 78.14314682167348
    - type: manhattan_spearman
      value: 78.57692322969135
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
      value: 79.16989925136942
    - type: cos_sim_spearman
      value: 76.5996225327091
    - type: euclidean_pearson
      value: 77.8319003279786
    - type: euclidean_spearman
      value: 76.42824009468998
    - type: manhattan_pearson
      value: 77.69118862737736
    - type: manhattan_spearman
      value: 76.25568104762812
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
      value: 87.42012286935325
    - type: cos_sim_spearman
      value: 88.15654297884122
    - type: euclidean_pearson
      value: 87.34082819427852
    - type: euclidean_spearman
      value: 88.06333589547084
    - type: manhattan_pearson
      value: 87.25115596784842
    - type: manhattan_spearman
      value: 87.9559927695203
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
      value: 82.88222044996712
    - type: cos_sim_spearman
      value: 84.28476589061077
    - type: euclidean_pearson
      value: 83.17399758058309
    - type: euclidean_spearman
      value: 83.85497357244542
    - type: manhattan_pearson
      value: 83.0308397703786
    - type: manhattan_spearman
      value: 83.71554539935046
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (ko-ko)
      type: mteb/sts17-crosslingual-sts
      config: ko-ko
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 80.20682986257339
    - type: cos_sim_spearman
      value: 79.94567120362092
    - type: euclidean_pearson
      value: 79.43122480368902
    - type: euclidean_spearman
      value: 79.94802077264987
    - type: manhattan_pearson
      value: 79.32653021527081
    - type: manhattan_spearman
      value: 79.80961146709178
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (ar-ar)
      type: mteb/sts17-crosslingual-sts
      config: ar-ar
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 74.46578144394383
    - type: cos_sim_spearman
      value: 74.52496637472179
    - type: euclidean_pearson
      value: 72.2903807076809
    - type: euclidean_spearman
      value: 73.55549359771645
    - type: manhattan_pearson
      value: 72.09324837709393
    - type: manhattan_spearman
      value: 73.36743103606581
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-ar)
      type: mteb/sts17-crosslingual-sts
      config: en-ar
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 71.37272335116
    - type: cos_sim_spearman
      value: 71.26702117766037
    - type: euclidean_pearson
      value: 67.114829954434
    - type: euclidean_spearman
      value: 66.37938893947761
    - type: manhattan_pearson
      value: 66.79688574095246
    - type: manhattan_spearman
      value: 66.17292828079667
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-de)
      type: mteb/sts17-crosslingual-sts
      config: en-de
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 80.61016770129092
    - type: cos_sim_spearman
      value: 82.08515426632214
    - type: euclidean_pearson
      value: 80.557340361131
    - type: euclidean_spearman
      value: 80.37585812266175
    - type: manhattan_pearson
      value: 80.6782873404285
    - type: manhattan_spearman
      value: 80.6678073032024
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
      value: 87.00150745350108
    - type: cos_sim_spearman
      value: 87.83441972211425
    - type: euclidean_pearson
      value: 87.94826702308792
    - type: euclidean_spearman
      value: 87.46143974860725
    - type: manhattan_pearson
      value: 87.97560344306105
    - type: manhattan_spearman
      value: 87.5267102829796
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (en-tr)
      type: mteb/sts17-crosslingual-sts
      config: en-tr
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 64.76325252267235
    - type: cos_sim_spearman
      value: 63.32615095463905
    - type: euclidean_pearson
      value: 64.07920669155716
    - type: euclidean_spearman
      value: 61.21409893072176
    - type: manhattan_pearson
      value: 64.26308625680016
    - type: manhattan_spearman
      value: 61.2438185254079
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (es-en)
      type: mteb/sts17-crosslingual-sts
      config: es-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 75.82644463022595
    - type: cos_sim_spearman
      value: 76.50381269945073
    - type: euclidean_pearson
      value: 75.1328548315934
    - type: euclidean_spearman
      value: 75.63761139408453
    - type: manhattan_pearson
      value: 75.18610101241407
    - type: manhattan_spearman
      value: 75.30669266354164
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (es-es)
      type: mteb/sts17-crosslingual-sts
      config: es-es
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 87.49994164686832
    - type: cos_sim_spearman
      value: 86.73743986245549
    - type: euclidean_pearson
      value: 86.8272894387145
    - type: euclidean_spearman
      value: 85.97608491000507
    - type: manhattan_pearson
      value: 86.74960140396779
    - type: manhattan_spearman
      value: 85.79285984190273
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (fr-en)
      type: mteb/sts17-crosslingual-sts
      config: fr-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 79.58172210788469
    - type: cos_sim_spearman
      value: 80.17516468334607
    - type: euclidean_pearson
      value: 77.56537843470504
    - type: euclidean_spearman
      value: 77.57264627395521
    - type: manhattan_pearson
      value: 78.09703521695943
    - type: manhattan_spearman
      value: 78.15942760916954
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (it-en)
      type: mteb/sts17-crosslingual-sts
      config: it-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 79.7589932931751
    - type: cos_sim_spearman
      value: 80.15210089028162
    - type: euclidean_pearson
      value: 77.54135223516057
    - type: euclidean_spearman
      value: 77.52697996368764
    - type: manhattan_pearson
      value: 77.65734439572518
    - type: manhattan_spearman
      value: 77.77702992016121
  - task:
      type: STS
    dataset:
      name: MTEB STS17 (nl-en)
      type: mteb/sts17-crosslingual-sts
      config: nl-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 79.16682365511267
    - type: cos_sim_spearman
      value: 79.25311267628506
    - type: euclidean_pearson
      value: 77.54882036762244
    - type: euclidean_spearman
      value: 77.33212935194827
    - type: manhattan_pearson
      value: 77.98405516064015
    - type: manhattan_spearman
      value: 77.85075717865719
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
      value: 59.10473294775917
    - type: cos_sim_spearman
      value: 61.82780474476838
    - type: euclidean_pearson
      value: 45.885111672377256
    - type: euclidean_spearman
      value: 56.88306351932454
    - type: manhattan_pearson
      value: 46.101218127323186
    - type: manhattan_spearman
      value: 56.80953694186333
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de)
      type: mteb/sts22-crosslingual-sts
      config: de
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 45.781923079584146
    - type: cos_sim_spearman
      value: 55.95098449691107
    - type: euclidean_pearson
      value: 25.4571031323205
    - type: euclidean_spearman
      value: 49.859978118078935
    - type: manhattan_pearson
      value: 25.624938455041384
    - type: manhattan_spearman
      value: 49.99546185049401
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (es)
      type: mteb/sts22-crosslingual-sts
      config: es
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 60.00618133997907
    - type: cos_sim_spearman
      value: 66.57896677718321
    - type: euclidean_pearson
      value: 42.60118466388821
    - type: euclidean_spearman
      value: 62.8210759715209
    - type: manhattan_pearson
      value: 42.63446860604094
    - type: manhattan_spearman
      value: 62.73803068925271
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (pl)
      type: mteb/sts22-crosslingual-sts
      config: pl
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 28.460759121626943
    - type: cos_sim_spearman
      value: 34.13459007469131
    - type: euclidean_pearson
      value: 6.0917739325525195
    - type: euclidean_spearman
      value: 27.9947262664867
    - type: manhattan_pearson
      value: 6.16877864169911
    - type: manhattan_spearman
      value: 28.00664163971514
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (tr)
      type: mteb/sts22-crosslingual-sts
      config: tr
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 57.42546621771696
    - type: cos_sim_spearman
      value: 63.699663168970474
    - type: euclidean_pearson
      value: 38.12085278789738
    - type: euclidean_spearman
      value: 58.12329140741536
    - type: manhattan_pearson
      value: 37.97364549443335
    - type: manhattan_spearman
      value: 57.81545502318733
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (ar)
      type: mteb/sts22-crosslingual-sts
      config: ar
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 46.82241380954213
    - type: cos_sim_spearman
      value: 57.86569456006391
    - type: euclidean_pearson
      value: 31.80480070178813
    - type: euclidean_spearman
      value: 52.484000620130104
    - type: manhattan_pearson
      value: 31.952708554646097
    - type: manhattan_spearman
      value: 52.8560972356195
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (ru)
      type: mteb/sts22-crosslingual-sts
      config: ru
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 52.00447170498087
    - type: cos_sim_spearman
      value: 60.664116225735164
    - type: euclidean_pearson
      value: 33.87382555421702
    - type: euclidean_spearman
      value: 55.74649067458667
    - type: manhattan_pearson
      value: 33.99117246759437
    - type: manhattan_spearman
      value: 55.98749034923899
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
      value: 58.06497233105448
    - type: cos_sim_spearman
      value: 65.62968801135676
    - type: euclidean_pearson
      value: 47.482076613243905
    - type: euclidean_spearman
      value: 62.65137791498299
    - type: manhattan_pearson
      value: 47.57052626104093
    - type: manhattan_spearman
      value: 62.436916516613294
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (fr)
      type: mteb/sts22-crosslingual-sts
      config: fr
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 70.49397298562575
    - type: cos_sim_spearman
      value: 74.79604041187868
    - type: euclidean_pearson
      value: 49.661891561317795
    - type: euclidean_spearman
      value: 70.31535537621006
    - type: manhattan_pearson
      value: 49.553715741850006
    - type: manhattan_spearman
      value: 70.24779344636806
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de-en)
      type: mteb/sts22-crosslingual-sts
      config: de-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 55.640574515348696
    - type: cos_sim_spearman
      value: 54.927959317689
    - type: euclidean_pearson
      value: 29.00139666967476
    - type: euclidean_spearman
      value: 41.86386566971605
    - type: manhattan_pearson
      value: 29.47411067730344
    - type: manhattan_spearman
      value: 42.337438424952786
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (es-en)
      type: mteb/sts22-crosslingual-sts
      config: es-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 68.14095292259312
    - type: cos_sim_spearman
      value: 73.99017581234789
    - type: euclidean_pearson
      value: 46.46304297872084
    - type: euclidean_spearman
      value: 60.91834114800041
    - type: manhattan_pearson
      value: 47.07072666338692
    - type: manhattan_spearman
      value: 61.70415727977926
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (it)
      type: mteb/sts22-crosslingual-sts
      config: it
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 73.27184653359575
    - type: cos_sim_spearman
      value: 77.76070252418626
    - type: euclidean_pearson
      value: 62.30586577544778
    - type: euclidean_spearman
      value: 75.14246629110978
    - type: manhattan_pearson
      value: 62.328196884927046
    - type: manhattan_spearman
      value: 75.1282792981433
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (pl-en)
      type: mteb/sts22-crosslingual-sts
      config: pl-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 71.59448528829957
    - type: cos_sim_spearman
      value: 70.37277734222123
    - type: euclidean_pearson
      value: 57.63145565721123
    - type: euclidean_spearman
      value: 66.10113048304427
    - type: manhattan_pearson
      value: 57.18897811586808
    - type: manhattan_spearman
      value: 66.5595511215901
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (zh-en)
      type: mteb/sts22-crosslingual-sts
      config: zh-en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 66.37520607720838
    - type: cos_sim_spearman
      value: 69.92282148997948
    - type: euclidean_pearson
      value: 40.55768770125291
    - type: euclidean_spearman
      value: 55.189128944669605
    - type: manhattan_pearson
      value: 41.03566433468883
    - type: manhattan_spearman
      value: 55.61251893174558
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (es-it)
      type: mteb/sts22-crosslingual-sts
      config: es-it
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 57.791929533771835
    - type: cos_sim_spearman
      value: 66.45819707662093
    - type: euclidean_pearson
      value: 39.03686018511092
    - type: euclidean_spearman
      value: 56.01282695640428
    - type: manhattan_pearson
      value: 38.91586623619632
    - type: manhattan_spearman
      value: 56.69394943612747
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de-fr)
      type: mteb/sts22-crosslingual-sts
      config: de-fr
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 47.82224468473866
    - type: cos_sim_spearman
      value: 59.467307194781164
    - type: euclidean_pearson
      value: 27.428459190256145
    - type: euclidean_spearman
      value: 60.83463107397519
    - type: manhattan_pearson
      value: 27.487391578496638
    - type: manhattan_spearman
      value: 61.281380460246496
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (de-pl)
      type: mteb/sts22-crosslingual-sts
      config: de-pl
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 16.306666792752644
    - type: cos_sim_spearman
      value: 39.35486427252405
    - type: euclidean_pearson
      value: -2.7887154897955435
    - type: euclidean_spearman
      value: 27.1296051831719
    - type: manhattan_pearson
      value: -3.202291270581297
    - type: manhattan_spearman
      value: 26.32895849218158
  - task:
      type: STS
    dataset:
      name: MTEB STS22 (fr-pl)
      type: mteb/sts22-crosslingual-sts
      config: fr-pl
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 59.67006803805076
    - type: cos_sim_spearman
      value: 73.24670207647144
    - type: euclidean_pearson
      value: 46.91884681500483
    - type: euclidean_spearman
      value: 16.903085094570333
    - type: manhattan_pearson
      value: 46.88391675325812
    - type: manhattan_spearman
      value: 28.17180849095055
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
      value: 83.79555591223837
    - type: cos_sim_spearman
      value: 85.63658602085185
    - type: euclidean_pearson
      value: 85.22080894037671
    - type: euclidean_spearman
      value: 85.54113580167038
    - type: manhattan_pearson
      value: 85.1639505960118
    - type: manhattan_spearman
      value: 85.43502665436196
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
      value: 80.73900991689766
    - type: mrr
      value: 94.81624131133934
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
      value: 55.678000000000004
    - type: map_at_10
      value: 65.135
    - type: map_at_100
      value: 65.824
    - type: map_at_1000
      value: 65.852
    - type: map_at_3
      value: 62.736000000000004
    - type: map_at_5
      value: 64.411
    - type: mrr_at_1
      value: 58.333
    - type: mrr_at_10
      value: 66.5
    - type: mrr_at_100
      value: 67.053
    - type: mrr_at_1000
      value: 67.08
    - type: mrr_at_3
      value: 64.944
    - type: mrr_at_5
      value: 65.89399999999999
    - type: ndcg_at_1
      value: 58.333
    - type: ndcg_at_10
      value: 69.34700000000001
    - type: ndcg_at_100
      value: 72.32
    - type: ndcg_at_1000
      value: 73.014
    - type: ndcg_at_3
      value: 65.578
    - type: ndcg_at_5
      value: 67.738
    - type: precision_at_1
      value: 58.333
    - type: precision_at_10
      value: 9.033
    - type: precision_at_100
      value: 1.0670000000000002
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 25.444
    - type: precision_at_5
      value: 16.933
    - type: recall_at_1
      value: 55.678000000000004
    - type: recall_at_10
      value: 80.72200000000001
    - type: recall_at_100
      value: 93.93299999999999
    - type: recall_at_1000
      value: 99.333
    - type: recall_at_3
      value: 70.783
    - type: recall_at_5
      value: 75.978
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
      value: 99.74653465346535
    - type: cos_sim_ap
      value: 93.01476369929063
    - type: cos_sim_f1
      value: 86.93009118541033
    - type: cos_sim_precision
      value: 88.09034907597535
    - type: cos_sim_recall
      value: 85.8
    - type: dot_accuracy
      value: 99.22970297029703
    - type: dot_ap
      value: 51.58725659485144
    - type: dot_f1
      value: 53.51351351351352
    - type: dot_precision
      value: 58.235294117647065
    - type: dot_recall
      value: 49.5
    - type: euclidean_accuracy
      value: 99.74356435643564
    - type: euclidean_ap
      value: 92.40332894384368
    - type: euclidean_f1
      value: 86.97838109602817
    - type: euclidean_precision
      value: 87.46208291203236
    - type: euclidean_recall
      value: 86.5
    - type: manhattan_accuracy
      value: 99.73069306930694
    - type: manhattan_ap
      value: 92.01320815721121
    - type: manhattan_f1
      value: 86.4135864135864
    - type: manhattan_precision
      value: 86.32734530938124
    - type: manhattan_recall
      value: 86.5
    - type: max_accuracy
      value: 99.74653465346535
    - type: max_ap
      value: 93.01476369929063
    - type: max_f1
      value: 86.97838109602817
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
      value: 55.2660514302523
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
      value: 30.4637783572547
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
      value: 49.41377758357637
    - type: mrr
      value: 50.138451213818854
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
      value: 28.887846011166594
    - type: cos_sim_spearman
      value: 30.10823258355903
    - type: dot_pearson
      value: 12.888049550236385
    - type: dot_spearman
      value: 12.827495903098123
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
      value: 1.667
    - type: map_at_100
      value: 9.15
    - type: map_at_1000
      value: 22.927
    - type: map_at_3
      value: 0.573
    - type: map_at_5
      value: 0.915
    - type: mrr_at_1
      value: 80
    - type: mrr_at_10
      value: 87.167
    - type: mrr_at_100
      value: 87.167
    - type: mrr_at_1000
      value: 87.167
    - type: mrr_at_3
      value: 85.667
    - type: mrr_at_5
      value: 87.167
    - type: ndcg_at_1
      value: 76
    - type: ndcg_at_10
      value: 69.757
    - type: ndcg_at_100
      value: 52.402
    - type: ndcg_at_1000
      value: 47.737
    - type: ndcg_at_3
      value: 71.866
    - type: ndcg_at_5
      value: 72.225
    - type: precision_at_1
      value: 80
    - type: precision_at_10
      value: 75
    - type: precision_at_100
      value: 53.959999999999994
    - type: precision_at_1000
      value: 21.568
    - type: precision_at_3
      value: 76.667
    - type: precision_at_5
      value: 78
    - type: recall_at_1
      value: 0.21
    - type: recall_at_10
      value: 1.9189999999999998
    - type: recall_at_100
      value: 12.589
    - type: recall_at_1000
      value: 45.312000000000005
    - type: recall_at_3
      value: 0.61
    - type: recall_at_5
      value: 1.019
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (sqi-eng)
      type: mteb/tatoeba-bitext-mining
      config: sqi-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.10000000000001
    - type: f1
      value: 90.06
    - type: precision
      value: 89.17333333333333
    - type: recall
      value: 92.10000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fry-eng)
      type: mteb/tatoeba-bitext-mining
      config: fry-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 56.06936416184971
    - type: f1
      value: 50.87508028259473
    - type: precision
      value: 48.97398843930635
    - type: recall
      value: 56.06936416184971
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kur-eng)
      type: mteb/tatoeba-bitext-mining
      config: kur-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 57.3170731707317
    - type: f1
      value: 52.96080139372822
    - type: precision
      value: 51.67861124382864
    - type: recall
      value: 57.3170731707317
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tur-eng)
      type: mteb/tatoeba-bitext-mining
      config: tur-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.3
    - type: f1
      value: 92.67333333333333
    - type: precision
      value: 91.90833333333333
    - type: recall
      value: 94.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (deu-eng)
      type: mteb/tatoeba-bitext-mining
      config: deu-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.7
    - type: f1
      value: 97.07333333333332
    - type: precision
      value: 96.79500000000002
    - type: recall
      value: 97.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nld-eng)
      type: mteb/tatoeba-bitext-mining
      config: nld-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.69999999999999
    - type: f1
      value: 93.2
    - type: precision
      value: 92.48333333333333
    - type: recall
      value: 94.69999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ron-eng)
      type: mteb/tatoeba-bitext-mining
      config: ron-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.9
    - type: f1
      value: 91.26666666666667
    - type: precision
      value: 90.59444444444445
    - type: recall
      value: 92.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ang-eng)
      type: mteb/tatoeba-bitext-mining
      config: ang-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 34.32835820895522
    - type: f1
      value: 29.074180380150533
    - type: precision
      value: 28.068207322920596
    - type: recall
      value: 34.32835820895522
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ido-eng)
      type: mteb/tatoeba-bitext-mining
      config: ido-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 78.5
    - type: f1
      value: 74.3945115995116
    - type: precision
      value: 72.82967843459222
    - type: recall
      value: 78.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (jav-eng)
      type: mteb/tatoeba-bitext-mining
      config: jav-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 66.34146341463415
    - type: f1
      value: 61.2469400518181
    - type: precision
      value: 59.63977756660683
    - type: recall
      value: 66.34146341463415
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (isl-eng)
      type: mteb/tatoeba-bitext-mining
      config: isl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 80.9
    - type: f1
      value: 76.90349206349207
    - type: precision
      value: 75.32921568627451
    - type: recall
      value: 80.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (slv-eng)
      type: mteb/tatoeba-bitext-mining
      config: slv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.93317132442284
    - type: f1
      value: 81.92519105034295
    - type: precision
      value: 80.71283920615635
    - type: recall
      value: 84.93317132442284
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cym-eng)
      type: mteb/tatoeba-bitext-mining
      config: cym-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 71.1304347826087
    - type: f1
      value: 65.22394755003451
    - type: precision
      value: 62.912422360248435
    - type: recall
      value: 71.1304347826087
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kaz-eng)
      type: mteb/tatoeba-bitext-mining
      config: kaz-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 79.82608695652173
    - type: f1
      value: 75.55693581780538
    - type: precision
      value: 73.79420289855072
    - type: recall
      value: 79.82608695652173
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (est-eng)
      type: mteb/tatoeba-bitext-mining
      config: est-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 74
    - type: f1
      value: 70.51022222222223
    - type: precision
      value: 69.29673599347512
    - type: recall
      value: 74
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (heb-eng)
      type: mteb/tatoeba-bitext-mining
      config: heb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 78.7
    - type: f1
      value: 74.14238095238095
    - type: precision
      value: 72.27214285714285
    - type: recall
      value: 78.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (gla-eng)
      type: mteb/tatoeba-bitext-mining
      config: gla-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 48.97466827503016
    - type: f1
      value: 43.080330405420874
    - type: precision
      value: 41.36505499593557
    - type: recall
      value: 48.97466827503016
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mar-eng)
      type: mteb/tatoeba-bitext-mining
      config: mar-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.60000000000001
    - type: f1
      value: 86.62333333333333
    - type: precision
      value: 85.225
    - type: recall
      value: 89.60000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lat-eng)
      type: mteb/tatoeba-bitext-mining
      config: lat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 45.2
    - type: f1
      value: 39.5761253006253
    - type: precision
      value: 37.991358436312
    - type: recall
      value: 45.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bel-eng)
      type: mteb/tatoeba-bitext-mining
      config: bel-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.5
    - type: f1
      value: 86.70333333333333
    - type: precision
      value: 85.53166666666667
    - type: recall
      value: 89.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pms-eng)
      type: mteb/tatoeba-bitext-mining
      config: pms-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 50.095238095238095
    - type: f1
      value: 44.60650460650461
    - type: precision
      value: 42.774116796477045
    - type: recall
      value: 50.095238095238095
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (gle-eng)
      type: mteb/tatoeba-bitext-mining
      config: gle-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 63.4
    - type: f1
      value: 58.35967261904762
    - type: precision
      value: 56.54857142857143
    - type: recall
      value: 63.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pes-eng)
      type: mteb/tatoeba-bitext-mining
      config: pes-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.2
    - type: f1
      value: 87.075
    - type: precision
      value: 86.12095238095239
    - type: recall
      value: 89.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nob-eng)
      type: mteb/tatoeba-bitext-mining
      config: nob-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.8
    - type: f1
      value: 95.90333333333334
    - type: precision
      value: 95.50833333333333
    - type: recall
      value: 96.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bul-eng)
      type: mteb/tatoeba-bitext-mining
      config: bul-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.9
    - type: f1
      value: 88.6288888888889
    - type: precision
      value: 87.61607142857142
    - type: recall
      value: 90.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cbk-eng)
      type: mteb/tatoeba-bitext-mining
      config: cbk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 65.2
    - type: f1
      value: 60.54377630539395
    - type: precision
      value: 58.89434482711381
    - type: recall
      value: 65.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hun-eng)
      type: mteb/tatoeba-bitext-mining
      config: hun-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87
    - type: f1
      value: 84.32412698412699
    - type: precision
      value: 83.25527777777778
    - type: recall
      value: 87
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (uig-eng)
      type: mteb/tatoeba-bitext-mining
      config: uig-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 68.7
    - type: f1
      value: 63.07883541295306
    - type: precision
      value: 61.06117424242426
    - type: recall
      value: 68.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (rus-eng)
      type: mteb/tatoeba-bitext-mining
      config: rus-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.7
    - type: f1
      value: 91.78333333333335
    - type: precision
      value: 90.86666666666667
    - type: recall
      value: 93.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (spa-eng)
      type: mteb/tatoeba-bitext-mining
      config: spa-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.7
    - type: f1
      value: 96.96666666666667
    - type: precision
      value: 96.61666666666667
    - type: recall
      value: 97.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hye-eng)
      type: mteb/tatoeba-bitext-mining
      config: hye-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.27493261455525
    - type: f1
      value: 85.90745732255168
    - type: precision
      value: 84.91389637616052
    - type: recall
      value: 88.27493261455525
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tel-eng)
      type: mteb/tatoeba-bitext-mining
      config: tel-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.5982905982906
    - type: f1
      value: 88.4900284900285
    - type: precision
      value: 87.57122507122507
    - type: recall
      value: 90.5982905982906
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (afr-eng)
      type: mteb/tatoeba-bitext-mining
      config: afr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.5
    - type: f1
      value: 86.90769841269842
    - type: precision
      value: 85.80178571428571
    - type: recall
      value: 89.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mon-eng)
      type: mteb/tatoeba-bitext-mining
      config: mon-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 82.5
    - type: f1
      value: 78.36796536796538
    - type: precision
      value: 76.82196969696969
    - type: recall
      value: 82.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (arz-eng)
      type: mteb/tatoeba-bitext-mining
      config: arz-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 71.48846960167715
    - type: f1
      value: 66.78771089148448
    - type: precision
      value: 64.98302885095339
    - type: recall
      value: 71.48846960167715
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hrv-eng)
      type: mteb/tatoeba-bitext-mining
      config: hrv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.1
    - type: f1
      value: 92.50333333333333
    - type: precision
      value: 91.77499999999999
    - type: recall
      value: 94.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nov-eng)
      type: mteb/tatoeba-bitext-mining
      config: nov-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 71.20622568093385
    - type: f1
      value: 66.83278891450098
    - type: precision
      value: 65.35065777283677
    - type: recall
      value: 71.20622568093385
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (gsw-eng)
      type: mteb/tatoeba-bitext-mining
      config: gsw-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 48.717948717948715
    - type: f1
      value: 43.53146853146853
    - type: precision
      value: 42.04721204721204
    - type: recall
      value: 48.717948717948715
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nds-eng)
      type: mteb/tatoeba-bitext-mining
      config: nds-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 58.5
    - type: f1
      value: 53.8564991863928
    - type: precision
      value: 52.40329436122275
    - type: recall
      value: 58.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ukr-eng)
      type: mteb/tatoeba-bitext-mining
      config: ukr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.8
    - type: f1
      value: 88.29
    - type: precision
      value: 87.09166666666667
    - type: recall
      value: 90.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (uzb-eng)
      type: mteb/tatoeba-bitext-mining
      config: uzb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 67.28971962616822
    - type: f1
      value: 62.63425307817832
    - type: precision
      value: 60.98065939771546
    - type: recall
      value: 67.28971962616822
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lit-eng)
      type: mteb/tatoeba-bitext-mining
      config: lit-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 78.7
    - type: f1
      value: 75.5264472455649
    - type: precision
      value: 74.38205086580086
    - type: recall
      value: 78.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ina-eng)
      type: mteb/tatoeba-bitext-mining
      config: ina-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.7
    - type: f1
      value: 86.10809523809525
    - type: precision
      value: 85.07602564102565
    - type: recall
      value: 88.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lfn-eng)
      type: mteb/tatoeba-bitext-mining
      config: lfn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 56.99999999999999
    - type: f1
      value: 52.85487521402737
    - type: precision
      value: 51.53985162713104
    - type: recall
      value: 56.99999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (zsm-eng)
      type: mteb/tatoeba-bitext-mining
      config: zsm-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94
    - type: f1
      value: 92.45333333333333
    - type: precision
      value: 91.79166666666667
    - type: recall
      value: 94
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ita-eng)
      type: mteb/tatoeba-bitext-mining
      config: ita-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.30000000000001
    - type: f1
      value: 90.61333333333333
    - type: precision
      value: 89.83333333333331
    - type: recall
      value: 92.30000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cmn-eng)
      type: mteb/tatoeba-bitext-mining
      config: cmn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.69999999999999
    - type: f1
      value: 93.34555555555555
    - type: precision
      value: 92.75416666666668
    - type: recall
      value: 94.69999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (lvs-eng)
      type: mteb/tatoeba-bitext-mining
      config: lvs-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 80.2
    - type: f1
      value: 76.6563035113035
    - type: precision
      value: 75.3014652014652
    - type: recall
      value: 80.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (glg-eng)
      type: mteb/tatoeba-bitext-mining
      config: glg-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.7
    - type: f1
      value: 82.78689263765207
    - type: precision
      value: 82.06705086580087
    - type: recall
      value: 84.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ceb-eng)
      type: mteb/tatoeba-bitext-mining
      config: ceb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 50.33333333333333
    - type: f1
      value: 45.461523661523664
    - type: precision
      value: 43.93545574795575
    - type: recall
      value: 50.33333333333333
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bre-eng)
      type: mteb/tatoeba-bitext-mining
      config: bre-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 6.6000000000000005
    - type: f1
      value: 5.442121400446441
    - type: precision
      value: 5.146630385487529
    - type: recall
      value: 6.6000000000000005
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ben-eng)
      type: mteb/tatoeba-bitext-mining
      config: ben-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85
    - type: f1
      value: 81.04666666666667
    - type: precision
      value: 79.25
    - type: recall
      value: 85
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (swg-eng)
      type: mteb/tatoeba-bitext-mining
      config: swg-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 47.32142857142857
    - type: f1
      value: 42.333333333333336
    - type: precision
      value: 40.69196428571429
    - type: recall
      value: 47.32142857142857
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (arq-eng)
      type: mteb/tatoeba-bitext-mining
      config: arq-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 30.735455543358945
    - type: f1
      value: 26.73616790022338
    - type: precision
      value: 25.397823220451283
    - type: recall
      value: 30.735455543358945
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kab-eng)
      type: mteb/tatoeba-bitext-mining
      config: kab-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 25.1
    - type: f1
      value: 21.975989896371022
    - type: precision
      value: 21.059885632257203
    - type: recall
      value: 25.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fra-eng)
      type: mteb/tatoeba-bitext-mining
      config: fra-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.3
    - type: f1
      value: 92.75666666666666
    - type: precision
      value: 92.06166666666665
    - type: recall
      value: 94.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (por-eng)
      type: mteb/tatoeba-bitext-mining
      config: por-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.1
    - type: f1
      value: 92.74
    - type: precision
      value: 92.09166666666667
    - type: recall
      value: 94.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tat-eng)
      type: mteb/tatoeba-bitext-mining
      config: tat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 71.3
    - type: f1
      value: 66.922442002442
    - type: precision
      value: 65.38249567099568
    - type: recall
      value: 71.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (oci-eng)
      type: mteb/tatoeba-bitext-mining
      config: oci-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 40.300000000000004
    - type: f1
      value: 35.78682789299971
    - type: precision
      value: 34.66425128716588
    - type: recall
      value: 40.300000000000004
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pol-eng)
      type: mteb/tatoeba-bitext-mining
      config: pol-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96
    - type: f1
      value: 94.82333333333334
    - type: precision
      value: 94.27833333333334
    - type: recall
      value: 96
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (war-eng)
      type: mteb/tatoeba-bitext-mining
      config: war-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 51.1
    - type: f1
      value: 47.179074753133584
    - type: precision
      value: 46.06461044702424
    - type: recall
      value: 51.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (aze-eng)
      type: mteb/tatoeba-bitext-mining
      config: aze-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.7
    - type: f1
      value: 84.71
    - type: precision
      value: 83.46166666666667
    - type: recall
      value: 87.7
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (vie-eng)
      type: mteb/tatoeba-bitext-mining
      config: vie-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.8
    - type: f1
      value: 94.68333333333334
    - type: precision
      value: 94.13333333333334
    - type: recall
      value: 95.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (nno-eng)
      type: mteb/tatoeba-bitext-mining
      config: nno-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.39999999999999
    - type: f1
      value: 82.5577380952381
    - type: precision
      value: 81.36833333333334
    - type: recall
      value: 85.39999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cha-eng)
      type: mteb/tatoeba-bitext-mining
      config: cha-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 21.16788321167883
    - type: f1
      value: 16.948865627297987
    - type: precision
      value: 15.971932568647897
    - type: recall
      value: 21.16788321167883
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mhr-eng)
      type: mteb/tatoeba-bitext-mining
      config: mhr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 6.9
    - type: f1
      value: 5.515526831658907
    - type: precision
      value: 5.141966366966367
    - type: recall
      value: 6.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (dan-eng)
      type: mteb/tatoeba-bitext-mining
      config: dan-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.2
    - type: f1
      value: 91.39666666666668
    - type: precision
      value: 90.58666666666667
    - type: recall
      value: 93.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ell-eng)
      type: mteb/tatoeba-bitext-mining
      config: ell-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.2
    - type: f1
      value: 89.95666666666666
    - type: precision
      value: 88.92833333333333
    - type: recall
      value: 92.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (amh-eng)
      type: mteb/tatoeba-bitext-mining
      config: amh-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 79.76190476190477
    - type: f1
      value: 74.93386243386244
    - type: precision
      value: 73.11011904761904
    - type: recall
      value: 79.76190476190477
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (pam-eng)
      type: mteb/tatoeba-bitext-mining
      config: pam-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 8.799999999999999
    - type: f1
      value: 6.921439712248537
    - type: precision
      value: 6.489885109680683
    - type: recall
      value: 8.799999999999999
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hsb-eng)
      type: mteb/tatoeba-bitext-mining
      config: hsb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 45.75569358178054
    - type: f1
      value: 40.34699501312631
    - type: precision
      value: 38.57886764719063
    - type: recall
      value: 45.75569358178054
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (srp-eng)
      type: mteb/tatoeba-bitext-mining
      config: srp-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.4
    - type: f1
      value: 89.08333333333333
    - type: precision
      value: 88.01666666666668
    - type: recall
      value: 91.4
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (epo-eng)
      type: mteb/tatoeba-bitext-mining
      config: epo-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.60000000000001
    - type: f1
      value: 92.06690476190477
    - type: precision
      value: 91.45095238095239
    - type: recall
      value: 93.60000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kzj-eng)
      type: mteb/tatoeba-bitext-mining
      config: kzj-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 7.5
    - type: f1
      value: 6.200363129378736
    - type: precision
      value: 5.89115314822466
    - type: recall
      value: 7.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (awa-eng)
      type: mteb/tatoeba-bitext-mining
      config: awa-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 73.59307359307358
    - type: f1
      value: 68.38933553219267
    - type: precision
      value: 66.62698412698413
    - type: recall
      value: 73.59307359307358
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fao-eng)
      type: mteb/tatoeba-bitext-mining
      config: fao-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 69.8473282442748
    - type: f1
      value: 64.72373682297346
    - type: precision
      value: 62.82834214131924
    - type: recall
      value: 69.8473282442748
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mal-eng)
      type: mteb/tatoeba-bitext-mining
      config: mal-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.5254730713246
    - type: f1
      value: 96.72489082969432
    - type: precision
      value: 96.33672974284326
    - type: recall
      value: 97.5254730713246
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ile-eng)
      type: mteb/tatoeba-bitext-mining
      config: ile-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 75.6
    - type: f1
      value: 72.42746031746033
    - type: precision
      value: 71.14036630036631
    - type: recall
      value: 75.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (bos-eng)
      type: mteb/tatoeba-bitext-mining
      config: bos-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.24293785310734
    - type: f1
      value: 88.86064030131826
    - type: precision
      value: 87.73540489642184
    - type: recall
      value: 91.24293785310734
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cor-eng)
      type: mteb/tatoeba-bitext-mining
      config: cor-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 6.2
    - type: f1
      value: 4.383083659794954
    - type: precision
      value: 4.027861324289673
    - type: recall
      value: 6.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (cat-eng)
      type: mteb/tatoeba-bitext-mining
      config: cat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.8
    - type: f1
      value: 84.09428571428572
    - type: precision
      value: 83.00333333333333
    - type: recall
      value: 86.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (eus-eng)
      type: mteb/tatoeba-bitext-mining
      config: eus-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 60.699999999999996
    - type: f1
      value: 56.1584972394755
    - type: precision
      value: 54.713456330903135
    - type: recall
      value: 60.699999999999996
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (yue-eng)
      type: mteb/tatoeba-bitext-mining
      config: yue-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.2
    - type: f1
      value: 80.66190476190475
    - type: precision
      value: 79.19690476190476
    - type: recall
      value: 84.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (swe-eng)
      type: mteb/tatoeba-bitext-mining
      config: swe-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.2
    - type: f1
      value: 91.33
    - type: precision
      value: 90.45
    - type: recall
      value: 93.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (dtp-eng)
      type: mteb/tatoeba-bitext-mining
      config: dtp-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 6.3
    - type: f1
      value: 5.126828976748276
    - type: precision
      value: 4.853614328966668
    - type: recall
      value: 6.3
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kat-eng)
      type: mteb/tatoeba-bitext-mining
      config: kat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 81.76943699731903
    - type: f1
      value: 77.82873739308057
    - type: precision
      value: 76.27622452019234
    - type: recall
      value: 81.76943699731903
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (jpn-eng)
      type: mteb/tatoeba-bitext-mining
      config: jpn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.30000000000001
    - type: f1
      value: 90.29666666666665
    - type: precision
      value: 89.40333333333334
    - type: recall
      value: 92.30000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (csb-eng)
      type: mteb/tatoeba-bitext-mining
      config: csb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 29.249011857707508
    - type: f1
      value: 24.561866096392947
    - type: precision
      value: 23.356583740215456
    - type: recall
      value: 29.249011857707508
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (xho-eng)
      type: mteb/tatoeba-bitext-mining
      config: xho-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.46478873239437
    - type: f1
      value: 73.23943661971832
    - type: precision
      value: 71.66666666666667
    - type: recall
      value: 77.46478873239437
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (orv-eng)
      type: mteb/tatoeba-bitext-mining
      config: orv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 20.35928143712575
    - type: f1
      value: 15.997867865075824
    - type: precision
      value: 14.882104658301346
    - type: recall
      value: 20.35928143712575
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ind-eng)
      type: mteb/tatoeba-bitext-mining
      config: ind-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.2
    - type: f1
      value: 90.25999999999999
    - type: precision
      value: 89.45333333333335
    - type: recall
      value: 92.2
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tuk-eng)
      type: mteb/tatoeba-bitext-mining
      config: tuk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 23.15270935960591
    - type: f1
      value: 19.65673625772148
    - type: precision
      value: 18.793705293464992
    - type: recall
      value: 23.15270935960591
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (max-eng)
      type: mteb/tatoeba-bitext-mining
      config: max-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 59.154929577464785
    - type: f1
      value: 52.3868463305083
    - type: precision
      value: 50.14938113529662
    - type: recall
      value: 59.154929577464785
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (swh-eng)
      type: mteb/tatoeba-bitext-mining
      config: swh-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 70.51282051282051
    - type: f1
      value: 66.8089133089133
    - type: precision
      value: 65.37645687645687
    - type: recall
      value: 70.51282051282051
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (hin-eng)
      type: mteb/tatoeba-bitext-mining
      config: hin-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.6
    - type: f1
      value: 93
    - type: precision
      value: 92.23333333333333
    - type: recall
      value: 94.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (dsb-eng)
      type: mteb/tatoeba-bitext-mining
      config: dsb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 38.62212943632568
    - type: f1
      value: 34.3278276962583
    - type: precision
      value: 33.07646935732408
    - type: recall
      value: 38.62212943632568
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ber-eng)
      type: mteb/tatoeba-bitext-mining
      config: ber-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 28.1
    - type: f1
      value: 23.579609223054604
    - type: precision
      value: 22.39622774921555
    - type: recall
      value: 28.1
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tam-eng)
      type: mteb/tatoeba-bitext-mining
      config: tam-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.27361563517914
    - type: f1
      value: 85.12486427795874
    - type: precision
      value: 83.71335504885994
    - type: recall
      value: 88.27361563517914
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (slk-eng)
      type: mteb/tatoeba-bitext-mining
      config: slk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.6
    - type: f1
      value: 86.39928571428571
    - type: precision
      value: 85.4947557997558
    - type: recall
      value: 88.6
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tgl-eng)
      type: mteb/tatoeba-bitext-mining
      config: tgl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.5
    - type: f1
      value: 83.77952380952381
    - type: precision
      value: 82.67602564102565
    - type: recall
      value: 86.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ast-eng)
      type: mteb/tatoeba-bitext-mining
      config: ast-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 79.52755905511812
    - type: f1
      value: 75.3055868016498
    - type: precision
      value: 73.81889763779527
    - type: recall
      value: 79.52755905511812
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (mkd-eng)
      type: mteb/tatoeba-bitext-mining
      config: mkd-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.9
    - type: f1
      value: 73.76261904761905
    - type: precision
      value: 72.11670995670995
    - type: recall
      value: 77.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (khm-eng)
      type: mteb/tatoeba-bitext-mining
      config: khm-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 53.8781163434903
    - type: f1
      value: 47.25804051288816
    - type: precision
      value: 45.0603482390186
    - type: recall
      value: 53.8781163434903
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ces-eng)
      type: mteb/tatoeba-bitext-mining
      config: ces-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.10000000000001
    - type: f1
      value: 88.88
    - type: precision
      value: 87.96333333333334
    - type: recall
      value: 91.10000000000001
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tzl-eng)
      type: mteb/tatoeba-bitext-mining
      config: tzl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 38.46153846153847
    - type: f1
      value: 34.43978243978244
    - type: precision
      value: 33.429487179487175
    - type: recall
      value: 38.46153846153847
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (urd-eng)
      type: mteb/tatoeba-bitext-mining
      config: urd-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.9
    - type: f1
      value: 86.19888888888887
    - type: precision
      value: 85.07440476190476
    - type: recall
      value: 88.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (ara-eng)
      type: mteb/tatoeba-bitext-mining
      config: ara-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.9
    - type: f1
      value: 82.58857142857143
    - type: precision
      value: 81.15666666666667
    - type: recall
      value: 85.9
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (kor-eng)
      type: mteb/tatoeba-bitext-mining
      config: kor-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.8
    - type: f1
      value: 83.36999999999999
    - type: precision
      value: 81.86833333333333
    - type: recall
      value: 86.8
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (yid-eng)
      type: mteb/tatoeba-bitext-mining
      config: yid-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 68.51415094339622
    - type: f1
      value: 63.195000099481234
    - type: precision
      value: 61.394033442972116
    - type: recall
      value: 68.51415094339622
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (fin-eng)
      type: mteb/tatoeba-bitext-mining
      config: fin-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.5
    - type: f1
      value: 86.14603174603175
    - type: precision
      value: 85.1162037037037
    - type: recall
      value: 88.5
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (tha-eng)
      type: mteb/tatoeba-bitext-mining
      config: tha-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.62043795620438
    - type: f1
      value: 94.40389294403892
    - type: precision
      value: 93.7956204379562
    - type: recall
      value: 95.62043795620438
  - task:
      type: BitextMining
    dataset:
      name: MTEB Tatoeba (wuu-eng)
      type: mteb/tatoeba-bitext-mining
      config: wuu-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 81.8
    - type: f1
      value: 78.6532178932179
    - type: precision
      value: 77.46348795840176
    - type: recall
      value: 81.8
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
      value: 2.603
    - type: map_at_10
      value: 8.5
    - type: map_at_100
      value: 12.985
    - type: map_at_1000
      value: 14.466999999999999
    - type: map_at_3
      value: 4.859999999999999
    - type: map_at_5
      value: 5.817
    - type: mrr_at_1
      value: 28.571
    - type: mrr_at_10
      value: 42.331
    - type: mrr_at_100
      value: 43.592999999999996
    - type: mrr_at_1000
      value: 43.592999999999996
    - type: mrr_at_3
      value: 38.435
    - type: mrr_at_5
      value: 39.966
    - type: ndcg_at_1
      value: 26.531
    - type: ndcg_at_10
      value: 21.353
    - type: ndcg_at_100
      value: 31.087999999999997
    - type: ndcg_at_1000
      value: 43.163000000000004
    - type: ndcg_at_3
      value: 22.999
    - type: ndcg_at_5
      value: 21.451
    - type: precision_at_1
      value: 28.571
    - type: precision_at_10
      value: 19.387999999999998
    - type: precision_at_100
      value: 6.265
    - type: precision_at_1000
      value: 1.4160000000000001
    - type: precision_at_3
      value: 24.490000000000002
    - type: precision_at_5
      value: 21.224
    - type: recall_at_1
      value: 2.603
    - type: recall_at_10
      value: 14.474
    - type: recall_at_100
      value: 40.287
    - type: recall_at_1000
      value: 76.606
    - type: recall_at_3
      value: 5.978
    - type: recall_at_5
      value: 7.819
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
      value: 69.7848
    - type: ap
      value: 13.661023167088224
    - type: f1
      value: 53.61686134460943
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
      value: 61.28183361629882
    - type: f1
      value: 61.55481034919965
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
      value: 35.972128420092396
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
      value: 85.59933241938367
    - type: cos_sim_ap
      value: 72.20760361208136
    - type: cos_sim_f1
      value: 66.4447731755424
    - type: cos_sim_precision
      value: 62.35539102267469
    - type: cos_sim_recall
      value: 71.10817941952506
    - type: dot_accuracy
      value: 78.98313166835548
    - type: dot_ap
      value: 44.492521645493795
    - type: dot_f1
      value: 45.814889336016094
    - type: dot_precision
      value: 37.02439024390244
    - type: dot_recall
      value: 60.07915567282321
    - type: euclidean_accuracy
      value: 85.3907134767837
    - type: euclidean_ap
      value: 71.53847289080343
    - type: euclidean_f1
      value: 65.95952206778834
    - type: euclidean_precision
      value: 61.31006346328196
    - type: euclidean_recall
      value: 71.37203166226914
    - type: manhattan_accuracy
      value: 85.40859510043511
    - type: manhattan_ap
      value: 71.49664104395515
    - type: manhattan_f1
      value: 65.98569969356485
    - type: manhattan_precision
      value: 63.928748144482924
    - type: manhattan_recall
      value: 68.17941952506597
    - type: max_accuracy
      value: 85.59933241938367
    - type: max_ap
      value: 72.20760361208136
    - type: max_f1
      value: 66.4447731755424
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
      value: 88.83261536073273
    - type: cos_sim_ap
      value: 85.48178133644264
    - type: cos_sim_f1
      value: 77.87816307403935
    - type: cos_sim_precision
      value: 75.88953021114926
    - type: cos_sim_recall
      value: 79.97382198952879
    - type: dot_accuracy
      value: 79.76287499514883
    - type: dot_ap
      value: 59.17438838475084
    - type: dot_f1
      value: 56.34566667855996
    - type: dot_precision
      value: 52.50349092359864
    - type: dot_recall
      value: 60.794579611949494
    - type: euclidean_accuracy
      value: 88.76857996662397
    - type: euclidean_ap
      value: 85.22764834359887
    - type: euclidean_f1
      value: 77.65379751543554
    - type: euclidean_precision
      value: 75.11152683839401
    - type: euclidean_recall
      value: 80.37419156144134
    - type: manhattan_accuracy
      value: 88.6987231730508
    - type: manhattan_ap
      value: 85.18907981724007
    - type: manhattan_f1
      value: 77.51967028849757
    - type: manhattan_precision
      value: 75.49992701795358
    - type: manhattan_recall
      value: 79.65044656606098
    - type: max_accuracy
      value: 88.83261536073273
    - type: max_ap
      value: 85.48178133644264
    - type: max_f1
      value: 77.87816307403935
---

## Multilingual-E5-base

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

This model has 12 layers and the embedding size is 768.

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


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: 南瓜的家常做法',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

## Supported Languages

This model is initialized from [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
and continually trained on a mixture of multilingual datasets.
It supports 100 languages from xlm-roberta,
but low-resource languages may see performance degradation.

## Training Details

**Initialization**: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)

**First stage**: contrastive pre-training with weak supervision

| Dataset                                                                                                | Weak supervision                      | # of text pairs |
|--------------------------------------------------------------------------------------------------------|---------------------------------------|-----------------|
| Filtered [mC4](https://huggingface.co/datasets/mc4)                                                    | (title, page content)                 | 1B              |
| [CC News](https://huggingface.co/datasets/intfloat/multilingual_cc_news)                               | (title, news content)                 | 400M            |
| [NLLB](https://huggingface.co/datasets/allenai/nllb)                                                   | translation pairs                     | 2.4B            |
| [Wikipedia](https://huggingface.co/datasets/intfloat/wikipedia)                                        | (hierarchical section title, passage) | 150M            |
| Filtered [Reddit](https://www.reddit.com/)                                                             | (comment, response)                   | 800M            |
| [S2ORC](https://github.com/allenai/s2orc)                                                              | (title, abstract) and citation pairs  | 100M            |
| [Stackexchange](https://stackexchange.com/)                                                            | (question, answer)                    | 50M             |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)                                                  | (input prompt, response)              | 80M             |
| [Miscellaneous unsupervised SBERT data](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | -                                     | 10M             |

**Second stage**: supervised fine-tuning

| Dataset                                                                                | Language     | # of text pairs |
|----------------------------------------------------------------------------------------|--------------|-----------------|
| [MS MARCO](https://microsoft.github.io/msmarco/)                                       | English      | 500k            |
| [NQ](https://github.com/facebookresearch/DPR)                                          | English      | 70k             |
| [Trivia QA](https://github.com/facebookresearch/DPR)                                   | English      | 60k             |
| [NLI from SimCSE](https://github.com/princeton-nlp/SimCSE)                             | English      | <300k           |
| [ELI5](https://huggingface.co/datasets/eli5)                                           | English      | 500k            |
| [DuReader Retrieval](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval) | Chinese      | 86k             |
| [KILT Fever](https://huggingface.co/datasets/kilt_tasks)                               | English      | 70k             |
| [KILT HotpotQA](https://huggingface.co/datasets/kilt_tasks)                            | English      | 70k             |
| [SQuAD](https://huggingface.co/datasets/squad)                                         | English      | 87k             |
| [Quora](https://huggingface.co/datasets/quora)                                         | English      | 150k            |
| [Mr. TyDi](https://huggingface.co/datasets/castorini/mr-tydi)                                                                           | 11 languages | 50k             |
| [MIRACL](https://huggingface.co/datasets/miracl/miracl)                                                                             | 16 languages | 40k             |

For all labeled datasets, we only use its training set for fine-tuning.

For other training details, please refer to our paper at [https://arxiv.org/pdf/2212.03533.pdf](https://arxiv.org/pdf/2212.03533.pdf).

## Benchmark Results on [Mr. TyDi](https://arxiv.org/abs/2108.08787)

| Model                 | Avg MRR@10 |       | ar   | bn | en | fi | id | ja | ko | ru | sw   | te | th |
|-----------------------|------------|-------|------| --- | --- | --- | --- | --- | --- | --- |------| --- | --- |
| BM25                  | 33.3       | | 36.7 | 41.3 | 15.1 | 28.8 | 38.2 | 21.7 | 28.1 | 32.9 | 39.6 | 42.4 | 41.7 |
| mDPR                  | 16.7       | | 26.0 | 25.8  | 16.2 | 11.3 | 14.6 | 18.1 | 21.9 | 18.5 | 7.3 | 10.6 | 13.5 |
| BM25 + mDPR           | 41.7       | | 49.1 | 53.5 | 28.4 | 36.5 | 45.5 | 35.5 | 36.2 | 42.7 | 40.5 | 42.0 | 49.2 |
|                       |            |
| multilingual-e5-small | 64.4       | | 71.5 | 66.3 | 54.5 | 57.7 | 63.2 | 55.4 | 54.3 | 60.8 | 65.4 | 89.1 | 70.1 |
| multilingual-e5-base  | 65.9       | | 72.3 | 65.0 | 58.5 | 60.8 | 64.9 | 56.6 | 55.8 | 62.7 | 69.0 | 86.6 | 72.7 |
| multilingual-e5-large | **70.5**   | | 77.5 | 73.2 | 60.8 | 66.8 | 68.5 | 62.5 | 61.6 | 65.8 | 72.7 | 90.2 | 76.2 |

## MTEB Benchmark Evaluation

Check out [unilm/e5](https://github.com/microsoft/unilm/tree/master/e5) to reproduce evaluation results 
on the [BEIR](https://arxiv.org/abs/2104.08663) and [MTEB benchmark](https://arxiv.org/abs/2210.07316).

## Support for Sentence Transformers

Below is an example for usage with sentence_transformers.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-base')
input_texts = [
    'query: how much protein should a female eat',
    'query: 南瓜的家常做法',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 i     s 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or traini     ng for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮     ,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,     放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油     锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀      6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
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

- Use "query: " prefix for symmetric tasks such as semantic similarity, bitext mining, paraphrase retrieval.

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

Long texts will be truncated to at most 512 tokens.
