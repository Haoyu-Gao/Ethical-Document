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

model_size = "small"
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


For more details, visit this page: https://github.com/persiannlp/parsinlu/ 
