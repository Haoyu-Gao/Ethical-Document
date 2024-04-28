---
language:
- ja
- en
license:
- mit
- openrail
tags:
- anime
- art
- stable-diffusion
- stable-diffusion-diffusers
- lora
- text-to-image
- diffusers
pipeline_tag: text-to-image
---

# ![Hotaru Jujo's LoRA Collection](header.webp)

- 十条蛍（Hotaru Jujo）の作成したLoRAを配布しています。  
  - You can download Hotaru Jujo's LoRA collection from this repo.
- [作者プロフィール / Author's profile](profile.md)
- すべてのLoRAは[MITライセンス](LICENSE)またはCreativeML Open RAIL-Mのデュアルライセンスでリリースされます。どちらかのライセンスを選択して使用できます。
  - All LoRA's are dual-licensed under [MIT LICENSE](LICENSE) or CreativeML Open RAIL-M.
- LoRAの使用にあたって事前承諾や事後報告などは一切必要ありませんが、TwitterなどSNSで紹介していただけると嬉しいです。  
  - No prior consent or after reporting is required for the use of LoRA, but I would appreciate it if you could introduce it on Twitter or other SNS.
- 配布中のLoRAは、特記していない限りCFG Scale 7、Clip skip 1を標準設定として開発・動作検証しています。
  - Unless otherwise noted, all LoRA's are developed and tested on "CFG Scale 7" and "Clip skip 1" settings.

## 目次 (Index)

[実験LoRA置き場 (Experimental LoRA files)](./experimental/README.md)

[アイコレクション](#アイコレクション-eye-collection) / 
[デフォル眼](#デフォル眼-comic-expressions) / [ジト目](#ジト目-comic-expression--scornful-eyes) / [白目](#白目-comic-expression--white-eyes) / [黒目](#黒目-comic-expression--black-eyes) / [(☆\_☆)／(♡\_♡)の目](#☆_☆／♡_♡の目-star-and-heart-shaped-eyes) / [オッドアイ固定化補助](#オッドアイ固定化補助-heterochromia-helper) / [あいうえお発音の口](#あいうえお発音の口-mouths-pronouncing-aiueo) / [官能的な表情](#官能的な表情-sensual-face) / [にやにやした表情の目と口](#にやにやした表情の目と口-smirking-eyes--slyly-mouth) / [デフォルメされた猫の目と口](#デフォルメされた猫の目と口-anime-cat-eyesmouth) / [猫の目＆猫の口](#猫の目＆猫の口-cat-eyes--cat-mouth) / [白い睫毛](#白い睫毛-white-eyelashes) / [極細の眼](#極細の眼-semi-closed-eyes) / [困り顔の眼](#困り顔の眼-worried-eyes) / [ドヤ顔](#ドヤ顔-doyagao--smug-showing-off-face) / [驚いた目＆眠そうな目](#驚いた目＆眠そうな目-surprised-eyes--sleepy-eyes) / [目隠れ](#目隠れ-hair-over-eyes) / [円形の口](#円形の口-circular-mouth) / [ぐにゃぐにゃ口](#ぐにゃぐにゃ口-wavy-mouth-set) / [閉じた口](#閉じた口-closed-mouth-set) / [口の大きさ変更](#口の大きさ変更-mouth-size-control) / [Hyper detailer / refiner / denoiser](#hyper-detailer--refiner--denoiser) / [前面ライトアップ](#前面ライトアップ-front-lighting) / [暗闇化／光る眼](#暗闇化／光る眼-darkness--glowing-eyes) / [2.5D変換](#25d変換-convert-2d-to-25d) / [ペーパーキャラクター](#ペーパーキャラクター-paper-character-effect) / [集中線](#集中線-comic-effect--concentrated-lines) / [コントラスト調整](#コントラスト調整-contrast-control) / [ぼかし＆背景ぼかし](#ぼかし＆背景ぼかし-blur--background-blur) / [キャラクター発光](#キャラクター発光-character-luminescence) / [トーンカーブ調整](#トーンカーブ調整-tone-curve-control) / [彩度調整](#彩度調整-saturation-control) / [ウィンク補助](#ウィンク補助-wink-helper) / [激おこ顔](#激おこ顔-extremely-angry-face) / [にっこり笑顔補助](#にっこり笑顔補助-smiling-face-helper) / [思案顔補助](#思案顔補助-thinking-face-helper) / [茹でダコ顔](#茹でダコ顔-strongly-embarrassed-face) / [青醒め顔](#青醒め顔-paled-face)

[Eye collection](#アイコレクション-eye-collection) / [Comic expressions](#デフォル眼-comic-expressions) / [Comic expression : scornful eyes](#ジト目-comic-expression--scornful-eyes) / [Comic expression : white eyes](#白目-comic-expression--white-eyes) / [Comic expression : black eyes](#黒目-comic-expression--black-eyes) / [Star and heart shaped eyes](#☆_☆／♡_♡の目-star-and-heart-shaped-eyes) / [Heterochromia helper](#オッドアイ固定化補助-heterochromia-helper) / [Mouths pronouncing A,I,U,E,O](#あいうえお発音の口-mouths-pronouncing-aiueo) / [Sensual face](#官能的な表情-sensual-face) / [Smirking eyes and mouth](#にやにやした表情の目と口-smirking-eyes--slyly-mouth)  / [Anime cat eyes/mouth](#デフォルメされた猫の目と口-anime-cat-eyesmouth) / [Cat eyes / Cat mouth](#猫の目＆猫の口-cat-eyes--cat-mouth) / [White eyelashes](#白い睫毛-white-eyelashes) / [Semi-closed eyes](#極細の眼-semi-closed-eyes) / [Worried eyes](#困り顔の眼-worried-eyes) / [Doyagao : smug, showing-off face](#ドヤ顔-doyagao--smug-showing-off-face) / [Surprised eyes / Sleepy eyes](#驚いた目＆眠そうな目-surprised-eyes--sleepy-eyes) / [Hair over eyes](#目隠れ-hair-over-eyes) / [Circular mouth](#円形の口-circular-mouth) / [Wavy mouth set](#ぐにゃぐにゃ口-wavy-mouth-set) / [Closed mouth set](#閉じた口-closed-mouth-set) / [Mouth size control](#口の大きさ変更-mouth-size-control) / [Hyper detailer / refiner / denoiser](#hyper-detailer--refiner--denoiser) / [Front lighting](#前面ライトアップ-front-lighting) / [Darkness / Glowing eyes](#暗闇化／光る眼-darkness--glowing-eyes) / [Convert 2D to 2.5D](#25d変換-convert-2d-to-25d) / [Paper character effect](#ペーパーキャラクター-paper-character-effect) / [Comic effect : concentrated lines](#集中線-comic-effect--concentrated-lines) / [Contrast control](#コントラスト調整-contrast-control) / [Blur / Background blur](#ぼかし＆背景ぼかし-blur--background-blur) / [Character luminescence](#キャラクター発光-character-luminescence) / [Tone curve control](#トーンカーブ調整-tone-curve-control) / [Saturation control](#彩度調整-saturation-control) / [Wink helper](#ウィンク補助-wink-helper) / [Extremely angry face](#激おこ顔-extremely-angry-face) / [Smiling face helper](#にっこり笑顔補助-smiling-face-helper) / [Thinking face helper](#思案顔補助-thinking-face-helper) / [Strongly embarrassed face](#茹でダコ顔-strongly-embarrassed-face) / [Paled face](#青醒め顔-paled-face)

-----------------------------------------------

## アイコレクション (Eye collection)

[詳しく見る／ダウンロード](./eyecolle/README.md)

[![Sample image](eyecolle/thumb.webp)](./eyecolle/README.md)

「アイコレクション」シリーズは、使用するデータモデルに依存することなく、いろいろな眼の形を再現できることを目的としたLoRA群です。

"Eye collection" is a series of LoRAs designed to reproduce various eye shapes without depending on data models.

## デフォル眼 (Comic expressions)

[詳しく見る／ダウンロード (Details/Download)](./comiceye/README.md)

[![Sample image](comiceye/thumb.webp)](./comiceye/README.md)
[![Sample image](comiceye/thumb2.webp)](./comiceye/README.md)

漫画・アニメ的なデフォルメ表現の眼を各種再現できます。

Deformation expressions which are familiar in manga and anime-style can be reproduced.

## ジト目 (Comic expression : scornful eyes)

[詳しく見る／ダウンロード (Details/Download)](./jitome/README.md)

[![Sample image 1](jitome/thumb1.webp)](./jitome/README.md) [![Sample image 2](jitome/thumb2.webp)](./jitome/README.md)

漫画・アニメ的なデフォルメ表現でおなじみ、ジト目を再現できます。

Many types of LoRA are available to reproduce scornful eyes, a familiar cartoon/anime deformation expression.

## 白目 (Comic expression : white eyes)

[詳しく見る／ダウンロード (Details/Download)](./whiteeyes/README.md)

[![Sample image](whiteeyes/thumb.webp)](./whiteeyes/README.md)

漫画・アニメ的なデフォルメ表現でおなじみ、白目を再現できるLoRAを各種用意しました。

Many types of LoRA are available to reproduce white eyes, a familiar cartoon/anime deformation expression.

## 黒目 (Comic expression : black eyes)

[詳しく見る／ダウンロード (Details/Download)](./blackeyes/README.md)

[![Sample image](blackeyes/thumb.webp)](./blackeyes/README.md)

漫画・アニメ的なデフォルメ表現でおなじみ、黒目を再現できるLoRAを6種類用意しました。

6 types of LoRA are available to reproduce black eyes(●_●), a familiar cartoon/anime deformation expression.

## (☆\_☆)／(♡\_♡)の目 (Star and heart shaped eyes)

[詳しく見る／ダウンロード (Details/Download)](./starhearteyes/README.md)

[![Sample image](starhearteyes/thumb.webp)](./starhearteyes/README.md)

漫画・アニメ的なデフォルメ表現でおなじみ、(☆\_☆)と(♡\_♡)の目を再現できます。

Star shaped and heart shaped eyes, familiar in manga and anime-style deformation expressions, can be reproduced.

## オッドアイ固定化補助 (Heterochromia helper)

[詳しく見る／ダウンロード (Details/Download)](./hetechro/README.md)

[![Sample image](hetechro/thumb.webp)](./hetechro/README.md)

オッドアイの色および左右の組み合わせを固定することができます。  
青・緑・黄・赤の4色、それぞれ左右の組み合わせで全12通りが用意されています。  
少し使い方に癖があるので、「使い方」を参照してください。

The color and left-right combination of the heterochromia eyes can be fixed.  
Total of 12 combinations of four colors (blue, green, yellow, and red), each with left and right sides, are available.  
There are a few quirks to using this LoRA. Please refer to the "Usage" section.

## あいうえお発音の口 (Mouths pronouncing A,I,U,E,O)

[詳しく見る／ダウンロード (Details/Download)](./talkmouth/README.md)

[![Sample image](talkmouth/thumb.webp)](./talkmouth/README.md)

「あ、い、う、え、お」の発声をしている形の口を再現できます。  
形に応じて他のさまざまな用途にも応用できます。

Reproduces mouths pronouncing Japanese 5 basic vowels, `"A" (Ah; /a/)` , `"I" (Ee; /i/)` , `"U" (Woo; /ɯ/)` , `"E" (Eh; /e/)` , `"O" (Oh; /o/)` .  
It can be applied to a variety of other applications depending on its shape.

## 官能的な表情 (Sensual face)

[詳しく見る／ダウンロード (Details/Download)](./sensualface/README.md)

[![Sample image](sensualface/thumb.webp)](./sensualface/README.md)

少しうるうるした半眼、ハの字型に下がり気味の眉毛、若干頬に赤みが差すなど、官能的な表情を再現できます。NSFWなシーンにも使えます。  
4種類を用意しました。

Reproduces sensual (voluptuous) face with half-closed (and a bit wet) eyes and inverted-v-shaped eyeblows. Also suitable for NSFW scenes.  
4 types are available.

## にやにやした表情の目と口 (Smirking eyes / Slyly mouth)

[詳しく見る／ダウンロード (Details/Download)](./smirking/README.md)

[![Sample image](smirking/thumb.webp)](./smirking/README.md)
[![Sample image](smirking/thumb_v100.webp)](./smirking/README.md)

にやにやした表情の目と口をそれぞれ再現できます。

Reproduces smirking eyes and slyly mouth.

## デフォルメされた猫の目と口 (Anime cat eyes/mouth)

[詳しく見る／ダウンロード (Details/Download)](./animecat/README.md)

[![Sample image](animecat/thumb.webp)](./animecat/README.md)

アニメ調にデフォルメされた猫の目、およびそれと組み合わせて使われる菱形の口を再現できます。

Reproduces anime cat eyes and rhombus shaped mouth.

## 猫の目＆猫の口 (Cat eyes / Cat mouth)

[詳しく見る／ダウンロード (Details/Download)](./cateyemouth/README.md)

[![Sample image](cateyemouth/thumb.webp)](./cateyemouth/README.md)

瞳孔が縦に細まる猫の目、およびω形の猫の口を再現できます。

Reproduces cat shaped (slit pupils) and cat-like shaped ("ω"-shaped) mouth.

## 白い睫毛 (White eyelashes)

[詳しく見る／ダウンロード (Details/Download)](./whiteeyelash/README.md)

[![Sample image](whiteeyelash/thumb.webp)](./whiteeyelash/README.md)

白髪／銀髪キャラの表現手法として使われることがある、白い睫毛を再現します。

Reproduces white eyelashes of white(silver)-hair character.

## 極細の眼 (Semi-closed eyes)

[詳しく見る／ダウンロード (Details/Download)](./hosome/README.md)

[![Sample image](hosome/thumb.webp)](./hosome/README.md)

閉じかけ、極細の眼を再現できます。マイナス適用すると広く開いた眼にもできます。  
細目キャラクターのほか、まばたきアニメーションの中間状態の作成にも使用できます。

Reproduces semi-closed (very thin) eyes, or widely open eyes (by minus LoRA weight).

## 困り顔の眼 (Worried eyes)

[詳しく見る／ダウンロード (Details/Download)](./worriedeyes/README.md)

[![Sample image](worriedeyes/thumb.webp)](./worriedeyes/README.md)

上瞼が谷型に曲がった、困り顔などで使われる目つきを再現できます。笑顔にも困り顔にも対応します。

Reproduces eyes with valley shaped eyelids, expressing worry, upset, confused, or thinking etc.

## ドヤ顔 (Doyagao : smug, showing-off face)

[詳しく見る／ダウンロード (Details/Download)](./doyagao/README.md)

[![Sample image](doyagao/thumb.webp)](./doyagao/README.md)

V字型眉のドヤ顔を再現できます。  
通常、V字型眉はV-shaped eyebrowsのプロンプトで再現できますが、たまに極太の眉毛になってしまうことがあります。そういった場合に、プロンプトの代わりにこのLoRAを使ってみてください。

Reproduces V-shaped eyebrows to express smug / proudly face (called "Doyagao" - Japanese anime slung).  
Usually, V-shaped eyebrows can be reproduced by using V-shaped eyebrows prompt, but it sometimes makes very thick eyebrows. This LoRA does not reproduce thick one.

## 驚いた目＆眠そうな目 (Surprised eyes / Sleepy eyes)

[詳しく見る／ダウンロード (Details/Download)](./sleepy_surprised/README.md)

[![Sample image](sleepy_surprised/thumb.webp)](./sleepy_surprised/README.md)

驚きに見開いた目、および眠そうな生気の無い半目を再現できます。

Reproduces wide-open surprised eyes or sleepy half-lidded eyes.

## 目隠れ (Hair over eyes)

[詳しく見る／ダウンロード (Details/Download)](./mekakure/README.md)

[![Sample image](mekakure/thumb.webp)](./mekakure/README.md)

前髪で目が隠れているキャラクターを再現できます。両目が隠れているパターンのほか、右側・左側の片目だけを隠した状態を再現するタイプも用意しました。

Reproduces character whose eyes are hidden by bangs. Three types are available : both eyes are hidden, right eye is hidden, or left eye is hidden.

## 円形の口 (Circular mouth)

[詳しく見る／ダウンロード (Details/Download)](./circlemouth/README.md)

[![Sample image](circlemouth/thumb.webp)](./circlemouth/README.md)

円形の口は`(:0)`のプロンプトで再現できますが、思ったより大きくなったり小さくなったりしてしまうことがあります。  
このLoRAを適用すると、大きいサイズまたは小さいサイズに固定することができます。

With most checkpoints, "o"-shaped (circular) mouth can be reproduced with prompt (:0), but its size may be larger or smaller than expected.  
With this LoRA, mouth size can be fixed to large size or small size.

## ぐにゃぐにゃ口 (Wavy mouth set)

[詳しく見る／ダウンロード (Details/Download)](./wavymouth/README.md)

[![Sample image](wavymouth/thumb.webp)](./wavymouth/README.md)

標準プロンプトで出せる`wavy mouth`の効果を拡張し、輪郭がぐにゃぐにゃした漫画的表現の口を生成することができます。  
形状別に6種類用意しました。

Extends `wavy mouth` prompt to produce a cartoon-like mouth with squishy contours.  
6 types of shapes are available.

## 閉じた口 (Closed mouth set)

[詳しく見る／ダウンロード (Details/Download)](./closedmouth/README.md)

[![Sample image](closedmouth/thumb.webp)](./closedmouth/README.md)

閉じた口の特殊な形を表現することができます。  
形の異なる2種類を公開しています。

Reproduces special shapes of the closed mouth.  
2 different types are available.

## 口の大きさ変更 (Mouth size control)

[詳しく見る／ダウンロード (Details/Download)](./widemouth/README.md)

[![Sample image](widemouth/thumb.webp)](./widemouth/README.md)

口の大きさを広げたり狭めたりすることができます。プラス適用すると大きく、マイナス適用すると小さくなります。  
形の異なる2種類を公開しています。

## Hyper detailer / refiner / denoiser

[詳しく見る／ダウンロード (Details/Download)](./hyperdetailer/README.md)

[![Sample image](hyperdetailer/thumb.webp)](./hyperdetailer/README.md)

出力画像の質感向上やディティールアップを行うLoRAを3種類公開しています。

Three LoRA's to detailing up or denoising.

## 前面ライトアップ (Front lighting)

[詳しく見る／ダウンロード (Details/Download)](./lightup/README.md)

[![Sample image](lightup/thumb.webp)](./lightup/README.md)

AIイラストでよく発生する「キャラクターの顔に影が落ちる」現象を改善するため、前面をライトアップできます。  
ライトアップ用のLoRAを使用すると塗りが甘くなりディティールが潰れる場合があるため、それを修正するディティールアップ用のLoRAも用意しています。

To improve the "shadow cast on the character's face" phenomenon that often occurs in AI illustrations, this LoRA lights up character's face.  
Since using "lighting up" LoRA may result painting and details to be poor, we also provide "detailing up" LoRA to use in combination.

## 暗闇化／光る眼 (Darkness / Glowing eyes)

[詳しく見る／ダウンロード (Details/Download)](./dark_gloweye/README.md)

[![Sample image](dark_gloweye/thumb.webp)](./dark_gloweye/README.md)

Stable Diffusionでキャラクターを出力すると、基本的にキャラクターの前側に光が当たった状態となり、暗い状態の再現が難しくなっています。  
このLoRAを使用すると、キャラクター前面にほとんど光が当たらない暗闇状態を再現しやすくなります。  
また、暗闇にいるキャラクターでよく演出として使用される「光る眼」を再現しやすくしたLoRAも同時に公開しています。

When using Stable Diffusion, basically front side of the character is lit up, making it difficult to reproduce a dark state.  
With this LoRA, it is easier to reproduce a dark state with almost no light on the front of the character.  
In addition, a LoRA is also available that makes it easier to reproduce the "glowing eyes" often used for characters in the dark as a dramatic effect.

## 2.5D変換 (Convert 2D to 2.5D)

[詳しく見る／ダウンロード (Details/Download)](./make25d/README.md)

[![Sample image](make25d/thumb.webp)](./make25d/README.md)

2Dアニメ系モデルの出力を、リアル／3D寄り（2.5D）な見た目に変換できます。

Converts output of 2D animated models to realistic/3D-like(2.5D) appearance.

## ペーパーキャラクター (Paper character effect)

[詳しく見る／ダウンロード (Details/Download)](./paperchara/README.md)

[![Sample image](paperchara/thumb.webp)](./paperchara/README.md)

アニメのおまけ映像などで見かける、キャラクターを紙に印刷して切り取ったような縁取りを付けた状態を再現できます。

Reproduces characters as printed on paper with a cut-out border, as seen in extra contents of some Japanese animations.

## 集中線 (Comic effect : concentrated lines)

[詳しく見る／ダウンロード (Details/Download)](./concentratedlines/README.md)

[![Sample image](concentratedlines/thumb.webp)](./concentratedlines/README.md)

背景に漫画的表現の集中線を出します。集中線のような形で色の付いたエフェクトになる場合も多いです。

Reproduces a concentrated line (mainly used in manga effect) in the background.

## コントラスト調整 (Contrast control)

[詳しく見る／ダウンロード (Details/Download)](./contrast/README.md)

[![Sample image](contrast/thumb.webp)](./contrast/README.md)

出力画像のコントラストを調整できます。  
通常の使用のほか、コントラストが高い／低いモデルにマージして出力品質を調整するといった用途に使うこともできます。

Adjust contrast of output images.  
It will be also usable to merge with low(or high)-contrast checkpoints to adjust default outputs.

## ぼかし＆背景ぼかし (Blur / Background blur)

[詳しく見る／ダウンロード (Details/Download)](./blur/README.md)

[![Sample image](blur/thumb.webp)](./blur/README.md)

blurは被写体含め全体を、blurbkは被写体を除いた背景部分だけを、ぼかしたりシャープにしたりすることができるエフェクトLoRAです。

You can blur or sharpen(and detail up) entire image or only background of output image. Minus weight makes sharpen effect.

## キャラクター発光 (Character luminescence)

[詳しく見る／ダウンロード (Details/Download)](./lumi/README.md)

[![Sample image](lumi/thumb.webp)](./lumi/README.md)

キャラクターの周囲に発光エフェクトを付与します。

Gives a luminescence effect around the character.

## トーンカーブ調整 (Tone curve control)

[詳しく見る／ダウンロード (Details/Download)](./tone/README.md)

[![Sample image](tone/thumb.webp)](./tone/README.md)

出力画像のトーンカーブを調整することができます。  
トーンアップ（白っぽくする）とトーンダウン（黒っぽくする）の2種類を用意しました。

Raises/Lowers the tone curve of the output image.

## 彩度調整 (Saturation control)

[詳しく見る／ダウンロード (Details/Download)](./saturation/README.md)

[![Sample image](saturation/thumb.webp)](./saturation/README.md)

出力画像の彩度をアップすることができます。テイスト別に3種類用意しました。

Increases saturation of output image. Three types are available.

## ウィンク補助 (Wink helper)

[詳しく見る／ダウンロード (Details/Download)](./wink/README.md)

[![Sample image](wink/thumb.webp)](./wink/README.md)

ウィンクをほぼ確実に出せるようになります。閉じる目を左右どちらにするか、LoRAを使い分けて指定できます。

## 激おこ顔 (Extremely angry face)

[詳しく見る／ダウンロード (Details/Download)](./gekioko/README.md)

[![Sample image](gekioko/thumb.webp)](./gekioko/README.md)

吊り上がった目で激しく怒っている、または不良のような表情を出すことができます。smileと合わせると不敵な笑みの表現にも使えます。  
雰囲気の異なる複数バージョンを公開しています。

## にっこり笑顔補助 (Smiling face helper)

[詳しく見る／ダウンロード (Details/Download)](./nikkori/README.md)

[![Sample image](nikkori/thumb.webp)](./nikkori/README.md)

閉じた目は`closed eyes`のプロンプトで再現できますが、形が悪かったりウィンクや半目になってしまうことがよくあります。  
このLoRAを使用すると、目を閉じてにっこりと笑っている目つきを安定して出すことができます。`closed eyes`のプロンプトだけよりも形状が整い、上向きに強めのカーブを描いた目になります。

To reproduce closed eyes, usually `closed eyes` prompt is used. But it may not certainly reproduce closed shape, sometimes get wink or half closed eyes.  
This LoRA helps to reproduce smiling faces with better shaped closed eyes. The eyes will have a stronger upward curve than the normal `closed eyes` prompt.

## 思案顔補助 (Thinking face helper)

[詳しく見る／ダウンロード (Details/Download)](./thinkingface/README.md)

[![Sample image](thinkingface/thumb.webp)](./thinkingface/README.md)

閉じた目は`closed eyes`のプロンプトで再現できますが、形が悪かったりウィンクや半目になってしまうことがよくあります。  
このLoRAを使用すると、目を閉じて考え込んでいる状態を安定して出すことができます。`closed eyes`のプロンプトだけよりも形状が整い、下向きに強めのカーブを描いた目になります。

To reproduce closed eyes, usually `closed eyes` prompt is used. But it may not certainly reproduce closed shape, sometimes get wink or half closed eyes.  
This LoRA helps to reproduce thoughtful look with better shaped closed eyes. The eyes will have a stronger downward curve than the normal `closed eyes` prompt.

## 茹でダコ顔 (Strongly embarrassed face)

[詳しく見る／ダウンロード (Details/Download)](./yudedako/README.md)

[![Sample image](yudedako/thumb.webp)](./yudedako/README.md)

俗に「茹でダコのような」などと呼ばれる、恥ずかしさで真っ赤になった顔を少しオーバー気味に表現できます。  
顔に赤線が入るタイプと、赤線が入らず赤く染まるだけのタイプ2種類（顔全体／頬のみ）の合計3種類を用意しました。  

Reproduces a face strongly turned red with embarrassment.  
Three types are available: one with a red line on the face, and two types with no red line but only a red tint (full face/cheeks only).

## 青醒め顔 (Paled face)

[詳しく見る／ダウンロード (Details/Download)](./paleface/README.md)

[![Details/Download](paleface/thumb.webp)](./paleface/README.md)

顔の上半分が青く染まる、恐怖や強い怒りなどをアニメチックに表現した顔を再現することができます。

Reproduces pale face (turn pale), an anime expression of fear or strong anger.

-----------------------------------------------

© 2023 Hotaru Jujo.

![Author's profile picture](profile.webp "This is a omage picture to a Japanese meme 'Kinuta dental clinic billboard'.")