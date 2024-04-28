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
**【告知】**  
**chilled_remix及びreversemixは2023年5月21日にVersion変更を行い、v2へ移行いたしました。**  
**伴いv1は削除致しました。なお既にDL済みの方は引き続き、v1をご利用いただくことは問題ございません。**  


License:[CreativeML Open RAIL-M](https://huggingface.co/sazyou-roukaku/chilled_remix/blob/main/license_v2.txt)<br>
Additional Copyright: sazyou_roukaku (TwitterID [@sazyou_roukaku](https://twitter.com/sazyou_roukaku)) as of May 21, 2023<br>

このモデルは『CreativeML Open RAIL-M』でLicenseそのものに変更はありません。<br>
~しかし追加著作者として鎖城郎郭の名前が追加されています。~<br>
しかし追加著作者として佐城郎画の名前が追加されています。(6/10 Twitterネーム変更に伴い、表記変更。License内はsazyou_roukakuの為変更なし)<br>
なお『CreativeML Open RAIL-M』に記載されている通り、<br>
本モデルを使用しての生成物に関してはLicenseの使用制限Aの事例を除き、当方は一切関与致しません。<br>
犯罪目的利用や医療用画像など特定専門的な用途での利用は使用制限Aで禁止されています。<br>
必ず確認しご利用ください。<br>
また当方は一切責任を持ちません。免責されていることをご了承の上、ご使用ください。<br>


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
      </tbody>
    </table>
  </div>
  なお、上記のモデルそのものの販売や商用画像生成サービスへの利用は、<br>
  『CreativeML Open RAIL-M』のLicense上、使用制限Aに追記記載しない限り、<br>
  制限することが本来できない為、マージ者への負担も考慮し、civitai制限表記上OKとしているだけであり、<br>
  積極的な推奨は行っておらず、またそれにより何らかの問題が生じても当方は一切責任を持ちません。<br>
  その点、ご留意いただくようお願いいたします。<br>
<br>

**推奨設定・モデルの違い・プロンプト**  

 Version2はfp16でVAE焼き込み版のみ配布といたしました。  
 基本的には**chilled_remixをメイン**とし、好みに合わせてreversemixも検討というのがスタンスです。  
 ※chilled_remixはchilled_re-genericユーザーをある騒動での混乱から守るために生み出されたモデルです。  
 性質上全てのユーザー出力に対応できなかった為、サブとしてreversemixが作られました。  
 reversemixはLORAなしでも顔のセミリアル感は薄いですが、全体的に幼くなる傾向があります。  


 chilled_remixはLORA愛用者の多いchilled_re-genericユーザー向けに生み出された為、    
 顔はLORAを使うとリアル感が一定になるよう設計されています。  
 プロンプトだけでもリアル化は可能ですが、LORAを少し使ってリアル化したほうが簡単です。  
 
 
 **CLIP設定:clip skip:2**を推奨。

 badhand系のnegativeTI無し、手系のネガティブも入れない出力と、  
 badhand系のnegativeTIを使った場合、正直大差ない感覚があります。  
 お好みでご利用ください。  

 自然言語的な文章プロンプトにかなり強いですが、シチュエーション以外の詳しい顔造形などは、  
 好みに合わせてワードプロンプトで指定するのが私のスタイルです。  
 ワードだけ構成でも問題があるわけではないので使いやすいスタイルで使ってください。  
 
 クオリティプロンプトは、high qualityなどは有効性を感じていません。  
 masterpieceは顔造形が変化する感覚ありますが、クオリティアップとしては微妙です。
 
 ただhigh resolutionは背景や質感に効果あります。high res、Hiresなど色々ありますが、  
 一番high resolutionを信頼しています。  


 
私が必ず入れるプロンプト  
(symmetrical clear eyes:1.3)は絶対入れてます。
目の色等や他の追加と合わせて分割したりもしますが、このプロンプト自体は入れるのをデフォルトとしています。  


愛用ネガティブプロンプトベース  
```
nipple,(manicure:1.2),(worst quality:2),(low quality:2),(long neck:2),(undressing:1.5),
```


**マージ利用モデル一覧**  
real-max-v3.4  
(https://civitai.com/models/60188/real-max-v34) ©dawn6666  
fantasticmix_v10(旧モデル名fantasticmixReal_v10)  
(https://civitai.com/models/22402/fantasticmixreal) ©michin  

dreamshaper_5Bakedvae  
(https://civitai.com/models/4384/dreamshaper) ©Lykon  
epicrealism_newAge  
(https://civitai.com/models/25694) ©epinikion  
diamondCoalMix_diamondCoalv2  
(https://civitai.com/models/41415) ©EnthusiastAI  


**FAQ**  
**Q1:何故v2を公開し、v1の配布を中止したのか**  
**A2:**  
 v1は元々マージ後も制限変更を禁止する表記になっているモデル（**realbiter_v10**）を使用していた為、  
 NG:Have different permissions when sharing mergesというcivitai制限を継承していました。  
 これは制限を追加することも解除することも不可という意味に取れます。一方でその他は全てOKでした。  
 つまり例えば  
 *NG:Sell this model or merges using this model*  
 *NG:Have different permissions when sharing merges*  
 こういうモデルとマージした時に**制限の矛盾**が発生し、**理屈上公開不可**という問題がありました。  

 マージをする者にとってこれは非常に厄介な制限で、また『CreativeML Open RAIL-M』にある  
 **Licenseを逸脱しない範囲であれば制限等を追加することができる**という文言にも抵触しています。  
 これが非常に気持ち悪く、嫌でした。  
 今回はその制限を解除する為のVersionアップです。  

 **v1の配布中止は、制限が異なる為、ややこしくトラブルの原因となる可能性がある点。**  
 
 また『CreativeML Open RAIL-M』には  
 **『更新に伴い、基本的に使用者は最新版を使う努力をすること』** の文面があります。  
 権利者は最新版を使わせるようにする権利を持ち、使用者は努力義務があるという内容です。  
 **ただし私はこの権利を行使致しませんので引き続きv1をお使いいただくことは問題ありません。**  
 しなしながらこの文面があるのに旧版を公開し続けるのは合理性に欠けることもあり、  
 誠に勝手ながら公開終了とさせていただきました。  
 ご理解のほどよろしくお願いいたします。  

 なおv1の再配布等は『CreativeML Open RAIL-M』に準拠致します。  



**Q2:今回の制限に問題や矛盾はないのか。**  
**A2:fantasticmix_v10**、**diamondCoalMix_diamondCoalv2**、**dreamshaper_5Bakedvae**は  
  **OK:Have different permissions when sharing merges**となっており解除可能。  
  **epicrealism_newAge**と**real-max-v3.4**は制限なしの為、今回全て制限なしとし公開しております。  

 なおマージ利用モデル側にLicense変更・制限変更等が生じた際も  
 5/17時点のLicenseや制限を前提として公開している為、creativeml-openrail-mに準じます。 
 こちらはMergeModel_LicenseSS_v2に該当モデルのSSを保管しております。  
  
 なおマージ利用モデル側に重大な問題が発生した場合は、モデルの公開停止を行い、  
 利用停止を呼びかける可能性はありますが、**当方側を理由とした追加制限を設けることは致しません。** 
<br>
<br>
<br>
<br>
<br>
<br>
**----------------------------下記は旧Version向け情報です------------------------**  
**chilled_remix_v1/chilled_reversemix_v1**に関して最低限の記載を残します。  
詳しい内容が必要な場合は編集履歴にて当時の記載をご確認ください。  
またMergeModel_LicenseSSに該当モデルの制限に関してSSを残しております。  

License:[CreativeML Open RAIL-M](https://huggingface.co/sazyou-roukaku/chilled_remix/blob/main/license.txt)<br>
Additional Copyright: sazyou_roukaku (TwitterID [@sazyou_roukaku](https://twitter.com/sazyou_roukaku)) as of April 18, 2023  

このモデルは『CreativeML Open RAIL-M』でLicenseそのものに変更はありません。  
しかし追加著作者として鎖城郎郭の名前が追加されています。  
なおcreativeml-openrail-mに記載されている通り、 本モデルを使用しての生成物に関しては使用制限Aの事例を除き、当方は一切関与致しません。  
また一切責任を持ちません。免責されていることをご了承の上、ご使用ください。  

**制限**
| Allowed | Permission                                          |
|:-------:|-----------------------------------------------------|
|    OK    | Use the model without crediting the creator        |
|    OK    | Sell images they generate                           |
|    OK    | Run on services that generate images for money      |
|    OK    | Share merges using this model                       |
|    OK    | Sell this model or merges using this model          |
|    NG    | Have different permissions when sharing merges      |
|          |                                                     |