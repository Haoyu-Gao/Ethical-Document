---
tags:
- text-to-image
- stable-diffusion
---

このモデルは以下の2ステップで作成されました。

1. [SSD-1B](https://huggingface.co/segmind/SSD-1B)を[NekorayXL](https://civitai.com/models/136719?modelVersionId=150826)と[sdxl-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)の差分の1.3倍でマージ。蒸留前と蒸留後のkeyについてはこの[マッピング](https://gist.github.com/laksjdjf/eddeda74a90ddaaaf4c51aea1ece7d01)を想定しています。
2. [NekorayXL](https://civitai.com/models/136719?modelVersionId=150826)の最終出力との差を損失にして蒸留（学習率1e-5,バッチサイズ4で23000ステップ)

# 使い方
[safetensors形式のファイル](https://huggingface.co/furusu/SSD-1B-anime/blob/main/ssd-1b-anime-v2.safetensors)は最新のComfyUIで使えます。

# LoRA
[ssd-1b-anime-cfgdistill](https://huggingface.co/furusu/SSD-1B-anime/blob/main/ssd-1b-anime-cfgdistill.safetensors):

cfg_scale=1でまともな画像が生成されるように学習したLoRAです。cfg_scale=1にするとネガティブ側の計算が必要なくなるため計算量が半分になります。1より大きくすると計算量削減の恩恵は受けられませんが、普通に性能向上LoRAとして使えるようです。ただし通常の生成よりは低い値をおすすめします。

# LCM

[lcm-ssd1b-anime](https://huggingface.co/furusu/SSD-1B-anime/blob/main/lcm-ssd1b-anime.safetensors):[SSD-1BのLCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b)から学習させたものです。



# SSD-1BとSDXLのkey対応について
[削除したモジュールがどれか分からないので](https://github.com/segmind/SSD-1B/issues/1)、コサイン類似度を利用して推定しました。
Transformer_depthだけ変わっているので（多分）Attention層のパラメータをSDXLとSSD-1B調査しました。
2層⇒1層となる場合先頭の層が残ります。
10層⇒4層となる場合1,2,3,7番目の層が残ります。

※up層の3番目は10層のままですが、コサイン類似度の結果が不可解なものになっていました。とりあえずここは変更されていないと仮定しています。


![image/png](https://cdn-uploads.huggingface.co/production/uploads/630591b9fca1d8d92b81bf02/JW84u7ZixzG5l_CyXiNqx.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/630591b9fca1d8d92b81bf02/lQz5gXmhMHkj81jAAzcJK.png)
