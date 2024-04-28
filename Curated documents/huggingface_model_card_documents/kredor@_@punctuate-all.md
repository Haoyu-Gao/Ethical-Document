---
{}
---
This is based on [Oliver Guhr's work](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large). The difference is that it is a finetuned xlm-roberta-base instead of an xlm-roberta-large and on twelve languages instead of four. The languages are: English, German, French, Spanish, Bulgarian, Italian, Polish, Dutch, Czech, Portugese, Slovak, Slovenian.

----- report -----

              precision    recall  f1-score   support

           0       0.99      0.99      0.99  73317475
           .       0.94      0.95      0.95   4484845
           ,       0.86      0.86      0.86   6100650
           ?       0.88      0.85      0.86    136479
           -       0.60      0.29      0.39    233630
           :       0.71      0.49      0.58    152424

    accuracy                           0.98  84425503
   macro avg       0.83      0.74      0.77  84425503
weighted avg       0.98      0.98      0.98  84425503


----- confusion matrix -----

     t/p      0     .     ,     ?     -     : 
        0   1.0   0.0   0.0   0.0   0.0   0.0 
        .   0.0   1.0   0.0   0.0   0.0   0.0 
        ,   0.1   0.0   0.9   0.0   0.0   0.0 
        ?   0.0   0.1   0.0   0.8   0.0   0.0 
        -   0.1   0.1   0.5   0.0   0.3   0.0 
        :   0.0   0.3   0.1   0.0   0.0   0.5 