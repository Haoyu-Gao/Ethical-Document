---
language:
- ja
license: creativeml-openrail-m
library_name: diffusers
tags:
- stable-diffusion
- text-to-image
pipeline_tag: text-to-image
---


License:[CreativeML Open RAIL-M](https://huggingface.co/sazyou-roukaku/BracingEvoMix/blob/main/license_v1.txt)<br>
Additional Copyright: sazyou_roukaku (TwitterID [@sazyou_roukaku](https://twitter.com/sazyou_roukaku)) as of May 31, 2023<br>

このモデルは『CreativeML Open RAIL-M』でLicenseそのものに変更はありません。<br>
~しかし追加著作者として鎖城郎郭の名前が追加されています。~<br>
しかし追加著作者として佐城郎画の名前が追加されています。(6/10 Twitterネーム変更に伴い、表記変更。License内はsazyou_roukakuの為変更なし)<br>
なお『CreativeML Open RAIL-M』に記載されている通り、<br>
本モデルを使用しての生成物に関してはLicenseの使用制限Aの事例を除き、当方は一切関与致しません。<br>
犯罪目的利用や医療用画像など特定専門的な用途での利用は使用制限Aで禁止されています。<br>
必ず確認しご利用ください。<br>
また当方は一切責任を持ちません。免責されていることをご了承の上、ご使用ください。<br>
<br>
2023/10/01<br>
BracingEvoMix_v2一般公開。<br>
<br>
**・BracingEvoMix_v2**<br>
CLIP設定/clip skip:2<br>
推奨ネガティブプロンプトベース:
```
(worst quality:2),(low quality:1,4),(undressing:1.5),(manicure:1.5),(long neck:2),lip,make up,(depth of field, bokeh, blurry, blurry background:1.4)
```
<br>
(depth of field, bokeh, blurry, blurry background:1.4)は背景を鮮明に出したいとき用です。<br>
<br>
なおBracingEvoMixシリーズの更新は一旦このv2にてストップします。<br>
追加学習を行っている方々たちがSDXLへ移行を進めており、選定基準に即したモデルで性能をあげるのに限界があること。<br>
下半身の一部ベースとなっているsxdがかなり初期のモデルの為、労力に対して、向上性能が誤差になる可能性が高いこと。<br>
利用可能なモデルの制限が厳しすぎて、服飾系の種類をいずれにせよ増やせないこと。<br>
以上が理由です。<br>
ただSD1.xの需要次第では、私自身でTrainingを行い、別シリーズを公開する可能性はあります。<br>
とはいえ、まずはSDXLでの動きや企業から使いやすいモデルが公開される可能性も見つつ、対応したいと考えています。<br>
<br>
<br>
<br>
  <h4>制限</h4>
  <div class="px-2">
    <table class="table-fixed border mt-0 text-xs">
        <tr>
          <td class="align-middle px-4 w-8">
            <span class="text-green-500">
              <h5>OK</h5>
            </span>
          </td>
          <td>
            著作者表記を入れずにモデルを使用する<br>
            Use the model without crediting the creator
          </td>
        </tr>
        <tr>
          <td class="align-middle px-4 w-8">
            <span class="text-green-500">
              <h5>OK</h5>
            </span>
          </td>
          <td>
            このモデルで生成した画像を商用利用する<br>
            Sell images they generate
          </td>
        </tr>
        <tr>
          <td class="align-middle px-4 w-8">
            <span class="text-green-500">
              <h5>OK</h5>
            </span>
          </td>
          <td>
            商用画像生成サービスに、このモデルを使用する<br>
            Run on services that generate images for money
          </td>
        </tr>
        <tr>
          <td class="align-middle px-4 w-8">
            <span class="text-green-500">
              <h5>OK</h5>
            </span>
          </td>
          <td>
            このモデルを使用したマージモデルを共有・配布する<br>
            Share merges using this model
          </td>
        </tr>
        <tr>
          <td class="align-middle px-4 w-8">
            <span class="text-green-500">
              <h5>OK</h5>
            </span>
          </td>
          <td>
            このモデル、または派生モデルを販売する<br>
            Sell this model or merges using this model
          </td>
        </tr>
        <tr>
          <td class="align-middle px-4 w-8">
            <span class="text-green-500">
              <h5>OK</h5>
            </span>
          </td>
          <td>
            このモデルをマージしたモデルに異なる権限を設定する<br>
            Have different permissions when sharing merges
          </td>
        </tr>
    </table>
  </div>
  なお、上記のモデルそのものの販売や商用画像生成サービスへの利用は、<br>
  『CreativeML Open RAIL-M』のLicense上、使用制限Aに追記記載しない限り、<br>
  制限することが本来できない為、マージ者への負担も考慮し、civitai制限表記上OKとしているだけであり、<br>
  積極的な推奨は行っておらず、またそれにより何らかの問題が生じても当方は一切責任を持ちません。<br>
  その点、ご留意いただくようお願いいたします。<br>
<br>


  <div class="px-2">
    <table class="table-fixed border mt-0 text-xs">
    <tr>
      <td colspan="2">
        <strong><h4>BracingEvoMix_v2</h4></strong>
      </td>
    </tr>
        <tr>
          <td>
            自然言語プロンプト反応(SD1.5)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>90点</h5>
          </td>
        </tr>
        <tr>
          <td>
            アジア顔出力の多様性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>95点</h5>
          </td>
        </tr>
        <tr>
          <td>
            フォトリアリティ<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>95点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            非現実的美形度<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>40点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            彩度・明度安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>65点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           手指の描画精度(最新上位モデルを90点とした場合)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>80点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           複雑ポーズ安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>75点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           乳部の意図せぬ露出制御性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           感情表現のプロンプト反応<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>70点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           表現可能年齢幅<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>85点</h5>
          </td>
  　　  </tr>
    <tr>
      <td colspan="2">
        <strong>基礎モデル『BracingEvoMix_v1』の正当後継版。彩度や明度のムラが改善され、指精度も上昇しています。
        また背景強化、横長解像度への強化が入っています。<br>
        出力ムラは減ったので汎用性は高くなったと思います。</strong><br>
      </td>
    </tr>
    <tr>
      <td colspan="2">
        <strong><h4>BracingEvoMix_v1</h4></strong>
      </td>
    </tr>
        <tr>
          <td>
            自然言語プロンプト反応(SD1.5)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>90点</h5>
          </td>
        </tr>
        <tr>
          <td>
            アジア顔出力の多様性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>95点</h5>
          </td>
        </tr>
        <tr>
          <td>
            フォトリアリティ<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>95点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            非現実的美形度<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>40点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            彩度・明度安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>40点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           手指の描画精度(最新上位モデルを90点とした場合)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>70点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           複雑ポーズ安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           乳部の意図せぬ露出制御性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>50点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           感情表現のプロンプト反応<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>80点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           表現可能年齢幅<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>85点</h5>
          </td>
  　　  </tr>
    <tr>
      <td colspan="2">
        <strong>基礎モデル。OpenBra由来のアジア顔を表現でき、リアリティも非常に高い反面、彩度や明度にムラがある。<br>
        感情表現にも強いので制御できれば表現幅は広い一方、弱点も多いじゃじゃ馬。<br>
        素材となった各モデルの良さと悪さを両方併せ持つので使いこなせれば、強いという玄人向け。</strong><br>
      </td>
    </tr>
    <tr>
      <td colspan="2">
        <strong><h4>BracingEvoMix_Another</h4></strong>
      </td>
    </tr>
        <tr>
          <td>
            自然言語プロンプト反応(SD1.5)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>90点</h5>
          </td>
        </tr>
        <tr>
          <td>
            アジア顔出力の多様性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>80点</h5>
          </td>
        </tr>
        <tr>
          <td>
            フォトリアリティ<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>90点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            非現実的美形度<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            彩度・明度安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>75点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           手指の描画精度(最新上位モデルを90点とした場合)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>85点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           複雑ポーズ安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           乳部の意図せぬ露出制御性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>55点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           感情表現のプロンプト反応<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>50点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           表現可能年齢幅<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
    　<tr>
      <td colspan="2">
        <strong>アナザーモデル。マージ手法を変更し、よりミキシングを深めることで彩度や明度の露骨な変化を安定化させた。<br>
        結果的にBRA顔が薄れ、少し一般的なアジア顔系マージモデルの方向性によっている。<br>
        指の安定性。特に爪や指先端の綺麗な表現はかなり最新モデル上位に比肩する。（グチャらない訳ではない）<br>
        弱みが減った分、感情表現反応も弱まった部分がある。なおlarge breasts以上での一般的な服装だとポロリのしやすさはそう大差なし。<br></strong>
      </td>
    　</tr>
  　　  </tr>
    <tr>
      <td colspan="2">
        <strong><h4>BracingEvoMix_Fast</h4></strong>
      </td>
    </tr>
        <tr>
          <td>
            自然言語プロンプト反応(SD1.5)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>90点</h5>
          </td>
        </tr>
        <tr>
          <td>
            アジア顔出力の多様性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>80点</h5>
          </td>
        </tr>
        <tr>
          <td>
            フォトリアリティ<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>80点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            非現実的美形度<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>65点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
            彩度・明度安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>75点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           手指の描画精度(最新上位モデルを90点とした場合)<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>80点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           複雑ポーズ安定性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           乳部の意図せぬ露出制御性<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           感情表現のプロンプト反応<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>50点</h5>
          </td>
  　　  </tr>
        <tr>
          <td>
           表現可能年齢幅<br>
          </td>
          <td class="align-middle px-4 w-15">
              <h5>60点</h5>
          </td>
    　<tr>
      <td colspan="2">
        <strong>アナザーモデルをよりchilled_remix的なアレンジをした試作モデル。<br>
        より完璧に整った顔立ちにしやすく、またエフェクト表現もchilled_remixほどではないも、多少通りやすいよう設計。<br>
        ただしフォトリアルの領域からは逸脱させていないので、chilled_remixほどの自由さはないです<br>
        </strong>
      </td>
    　</tr>
  　　  </tr>
    </table>
  </div>


**マージ利用モデル一覧**  
**[BracingEvoMix_v2]**  
OpenBra  
**©BanKai** [@PleaseBanKai](https://twitter.com/PleaseBanKai)  
dreamshaper_6BakedVae  
(https://civitai.com/models/4384) ©Lykon  
epicrealism_newEra  
epicrealism_pureEvolutionV5  
(https://civitai.com/models/25694) ©epinikion  
diamondCoalMix_diamondCoalv2  
(https://civitai.com/models/41415) ©EnthusiastAI  
sxd_v10  
(https://civitai.com/models/1169) ©izuek  
Evt_V4_e04_ema  
(https://huggingface.co/haor/Evt_V4-preview) ©haor  
bp_mk5  
(https://huggingface.co/Crosstyan/BPModel) ©Crosstyan  

**[BracingEvoMix_v1]**  
OpenBraβ  
OpenBra  
**©BanKai** [@PleaseBanKai](https://twitter.com/PleaseBanKai)  

dreamshaper_5Bakedvae  
dreamshaper_6BakedVae  
(https://civitai.com/models/4384) ©Lykon  
epicrealism_newAge  
epicrealism_newEra  
(https://civitai.com/models/25694) ©epinikion  
diamondCoalMix_diamondCoalv2  
(https://civitai.com/models/41415) ©EnthusiastAI  
sxd_v10  
(https://civitai.com/models/1169) ©izuek    
Evt_V4_e04_ema  
(https://huggingface.co/haor/Evt_V4-preview) ©haor  

**[BracingEvoMix_Another]**  
OpenBra  
**©BanKai** [@PleaseBanKai](https://twitter.com/PleaseBanKai)  
dreamshaper_6BakedVae  
(https://civitai.com/models/4384) ©Lykon  
epicrealism_newEra  
(https://civitai.com/models/25694) ©epinikion  
sxd_v10  
(https://civitai.com/models/1169) ©izuek    
Evt_V4_e04_ema  
(https://huggingface.co/haor/Evt_V4-preview) ©haor  

**[BracingEvoMix_Fast]**  
OpenBra  
**©BanKai** [@PleaseBanKai](https://twitter.com/PleaseBanKai)  
dreamshaper_6BakedVae  
(https://civitai.com/models/4384) ©Lykon  
epicrealism_newEra  
(https://civitai.com/models/25694) ©epinikion  
Evt_V4_e04_ema  
(https://huggingface.co/haor/Evt_V4-preview) ©haor  
bp_mk5  
(https://huggingface.co/Crosstyan/BPModel) ©Crosstyan  

--------------------------------------------------------------------------
**推奨設定**  
 
 
 CLIP設定/clip skip:2

 badhand系のnegativeTIを入れた場合と入れない場合の差は感覚的に大差ありません。  
 このモデルは**EasyNegative**もしくは**BadBras推奨**です。
 **EasyNegative v2** は人体構造破綻率が上記と比べて高いので非推奨。

 自然言語的な文章プロンプトにかなり強いですが、シチュエーション以外の詳しい顔造形などは、  
 好みに合わせてワードプロンプトで指定するのが私のスタイルです。  
 ワードだけ構成でも問題があるわけではないので使いやすいスタイルで使ってください。  
 
 クオリティプロンプトは、high qualityなどは有効性を感じていません。  
 masterpieceは顔造形が変化する感覚ありますが、クオリティアップとしては微妙です。
 
 ただhigh resolutionは背景や質感に効果あります。high res、Hiresなど色々ありますが、  
 一番high resolutionを信頼しています。  

 また1girlなどのWD式プロンプトではなく、girl、womanやyoung woman等の自然言語での使用も検討ください。  
 CGFスケールを少し下げ、5程度にすることで、全体的なコントラストを弱めることも技法として有用です。  
 **肌の色が濃く感じる場合は(fair skin:1.2)を入れると白くなります。上記と併せてご利用ください**

ネガティブプロンプトベース  
```
EasyNegative,(worst quality:2),(low quality:1.4),(undressing:1.5), (disheveled clothes:1.4),(manicure:1.2),(nipple:1.2),(long neck:2),
```


**FAQ**  
**Q1:BracingEvoMixとは何か**  
**A1:**  
従来のマージモデルはNAIleakモデルの混入やDreamlikeLicenseの混入の恐れがあり、<br>
本格的なビジネス利用において判断に困るような場面が見受けられました。<br>
今回のBracingEvoMixは、BRAの学習開発者であるBanKai氏と直接話し合い、<br>
有志の寄付の結果生まれたOpenBraβ・OpenBraをベースにマージしたモデルです。<br>
他も全て学習モデルとなっており、マージモデルでの組み合わせよりリスクモデルの混入確率をグンと減らしています。<br>
<br>
極東アジア系の顔が出るマージモデルの中では一番低リスクモデルだと考えられます。<br>
しかし学習モデルと名乗っていても、その中身に関して詳細を知ることはできないのでリスクゼロではありません。<br>
コサイン一致率などを元に選定も行っておりますが、混入がないとは言い切れない点はご承知おきください。<br>
<br>
<br>
**Q2:完全な混入していないモデルは無理なのか**  
**A2:**  
追加学習のみだと、データの偏りが生まれ、背景などを含めてどうしても能力にムラが出ます。<br>
個人レベルで完全な学習モデルを生み出すのは不可能に近い為、マージで対応する必要があります。<br>
しかしこれらの学習モデルの中身に関しては専門家でもないので、解析できません。<br>
<br>
なお差分抽出にて、特定モデルのデータと一致するデータを除去することは可能ですが、<br>
その差を取ることそのものが、派生モデルとしての条件を満たす可能性があります。<br>
またNAIleakモデル本体をDLしなければならず、そこも含めて困難と言えます。<br>
<br>
**Q3:各学習モデル選定基準について**  
**A3:**  
①sxd_v10<br>
これはNSFWモデルです。NSFWモデルを入れると脱衣をしやすくなるなど問題がある反面、<br>
OpenBraの弱点の一つである下半身の学習度合がやや低い点の補強と様々なポーズ学習データ補強に用いています。<br>
なお割合を増やし過ぎると露出がしやすくなる為、最低限の量をマージしています。<br>
sxdは比較的初期モデルながら、学習量も多めで、制限もない為、選択しています。<br>
<br>
②dreamshaper_5Bakedvae/dreamshaper_6BakedVae<br>
dreamshaperは商用画像生成サービスでも利用されている有名な学習モデルです。<br>
学習モデルとして追加学習量も多くデータの多様性も高い為、多様性補強用に採用しています。<br>
<br>
③epicrealism_newAge/epicrealism_newEra<br>
現行の学習モデルで最強のスペックを誇ると思われます。背景補強と能力の高さから採用しています。<br>
<br>
④diamondcoalmix_diamondcoalv2<br>
極東アジア人顔の系統データを持つ学習モデル。作者がLORA作成を多く行っており、またコサイン値からも問題ないと判断し、極東アジアデータ補強で採用。<br>
<br>
⑤Evt_V4_e04_ema<br>
ACertaintyというNAIleakデータを含まないと公言しているイラスト学習モデルでトレーニングを行い生み出されたモデル。<br>
イラストモデルをマージすることで、人種・年齢関係なく、顔データに影響を与えられるため、<br>
肖像権侵害率を下げる意味合いと日本人好みの顔立ちにする為に採用しています。<br>
<br>
※ACertaintyはNOVEL AIのデータを蒸留している可能性はありますが、こちらは特許法に抵触しない為、問題ないと考えています。<br>
ACertainty<br>
https://huggingface.co/JosephusCheung/ACertainty<br>
https://huggingface.co/JosephusCheung/ASimilarityCalculatior<br>
<br>
⑥bp_mk5<br>
ACertaintyベースの学習モデル。BracingEvoMix_Fastはリバランスの為に用いています。
<br>
<br>
**Q4:無印の指のクオリティがchilled_remix等と比べて低い**  
**A4:**  
使用モデルが一定の基準をクリアした学習モデルに限定されているため、データの安定性が最新モデルと比べて低いです。<br>
それでも初期モデルよりはまだシャープな指として出力され、3月中頃のモデルと比較して、平均的な出力だと感じています。<br>
~今後可能ならば、指強化等も検討しますが学習データからして困難と見ています。~<br>
申し訳ございませんが、controlnetでの制御などもご検討ください。<br>
<br>
6/14 追記:AnotherやFastはこの指の精度改善を行い、6月時点の最新フォトモデルの平均並みの出力だと考えています。

**Q5:今回の制限に問題や矛盾はないのか**  
**A5:** **diamondCoalMix_diamondCoalv2** 、 **dreamshaper_5Bakedvae** 、 **dreamshaper_6BakedVae** は  
  **OK:Have different permissions when sharing merges**となっており解除可能。  
  他は制限なしの為、今回全て制限なしとし公開しております。  

 なおマージ利用モデル側にLicense変更・制限変更等が生じた際も  
 5/31時点のLicenseや制限を前提として公開している為、creativeml-openrail-mに準じます。 
 こちらはMergeModel_LicenseSS_v1に該当モデルのSSを保管しております。  
  
 なおマージ利用モデル側に重大な問題が発生した場合は、モデルの公開停止を行い、  
 利用停止を呼びかける可能性はありますが、**当方側を理由とした追加制限を設けることは致しません。** 