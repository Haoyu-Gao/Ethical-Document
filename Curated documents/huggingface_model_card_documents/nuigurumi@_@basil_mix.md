---
license: other
---
## Model Description

- merged model.
- realistic texture and Asian face.
- designed to maintain a responsive reaction to danbooru based prompts.

## License
  
- This model and its derivatives(image, merged model) can be freely used for non-profit purposes only.
- You may not use this model and its derivatives on websites, apps, or other platforms where you can or plan to earn income or donations. If you wish to use it for such purposes, please contact nuigurumi.
- Introducing the model itself is allowed for both commercial and non-commercial purposes, but please include the model name and a link to this repository when doing so.

- このモデル及びその派生物(生成物、マージモデル)は、完全に非営利目的の使用に限り、自由に利用することができます。
- あなたが収入や寄付を得ることのできる、もしくは得る予定のWebサイト、アプリ、その他でこのモデル及びその派生物を利用することはできません。利用したい場合は[nuigurumi](https://twitter.com/nuigurumi1_KR)に連絡してください。
- モデル自体の紹介することは、営利非営利を問わず自由です、その場合はモデル名と当リポジトリのリンクを併記してください。

- check [License](https://huggingface.co/nuigurumi/basil_mix/blob/main/License.md)
  
  
  _読むのめんどくさい人向け  
  商用利用をすべて禁止します。fanboxやpatreonなどの支援サイトでの使用も全て禁止します。  
  マージモデル(cilled_re...とか)も派生物なので商用利用禁止になります。 商用利用をしたいなら私に連絡してください。  
  どこかでモデルを紹介していただけるなら、リンクも併記してくれると嬉しいです。_ 

# Gradio

We support a [Gradio](https://github.com/gradio-app/gradio) Web UI to run basil_mix:
[![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/akhaliq/basil_mix)


## Recommendations

- VAE: [vae-ft-mse-840000](https://huggingface.co/stabilityai/sd-vae-ft-mse-original) from StabilityAI
- Prompting: Simple prompts are better. Large amounts of quality tags and negative prompts can have negative effects.