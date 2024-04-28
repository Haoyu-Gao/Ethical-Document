---
language:
- fa
- multilingual
license: cc-by-nc-sa-4.0
tags:
- machine-translation
- mt5
- persian
- farsi
datasets:
- parsinlu
metrics:
- sacrebleu
thumbnail: https://upload.wikimedia.org/wikipedia/commons/a/a2/Farsi.svg
---

# Machine Translation (ترجمه‌ی ماشینی)

This is an mT5-based model for machine translation (Persian -> English). 
Here is an example of how you can run this model: 

```python 
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_size = "base"
model_name = f"persiannlp/mt5-{model_size}-parsinlu-opus-translation_fa_en"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("ستایش خدای را که پروردگار جهانیان است.")
run_model("در هاید پارک کرنر بر گلدانی ایستاده موعظه می‌کند؛")
run_model("وی از تمامی بلاگرها، سازمان‌ها و افرادی که از وی پشتیبانی کرده‌اند، تشکر کرد.")
run_model("مشابه سال ۲۰۰۱، تولید آمونیاک بی آب در ایالات متحده در سال ۲۰۰۰ تقریباً ۱۷،۴۰۰،۰۰۰ تن (معادل بدون آب) با مصرف ظاهری ۲۲،۰۰۰،۰۰۰ تن و حدود ۴۶۰۰۰۰۰ با واردات خالص مواجه شد. ")
run_model("می خواهم دکترای علوم کامپیوتر راجع به شبکه های اجتماعی را دنبال کنم، چالش حل نشده در شبکه های اجتماعی چیست؟")
```

which should give the following: 
```
['the admiration of God, which is the Lord of the world.']
['At the Ford Park, the Crawford Park stands on a vase;']
['He thanked all the bloggers, the organizations, and the people who supported him']
['similar to the year 2001, the economy of ammonia in the United States in the']
['I want to follow the computer experts on social networks, what is the unsolved problem in']
```

which should give the following: 
```
['Adoration of God, the Lord of the world.']
['At the High End of the Park, Conrad stands on a vase preaching;']
['She thanked all the bloggers, organizations, and men who had supported her.']
['In 2000, the lack of water ammonia in the United States was almost']
['I want to follow the computer science doctorate on social networks. What is the unsolved challenge']
```

Which should produce the following: 
```
['the praise of God, the Lord of the world.']
['At the Hyde Park Corner, Carpenter is preaching on a vase;']
['He thanked all the bloggers, organizations, and people who had supported him.']
['Similarly in 2001, the production of waterless ammonia in the United States was']
['I want to pursue my degree in Computer Science on social networks, what is the']
```


For more details, visit this page: https://github.com/persiannlp/parsinlu/ 
