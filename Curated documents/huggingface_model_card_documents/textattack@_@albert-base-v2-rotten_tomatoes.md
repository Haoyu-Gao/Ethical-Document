---
{}
---
## albert-base-v2 fine-tuned with TextAttack on the rotten_tomatoes dataset
    
    This `albert-base-v2` model was fine-tuned for sequence classificationusing TextAttack 
    and the rotten_tomatoes dataset loaded using the `nlp` library. The model was fine-tuned 
    for 10 epochs with a batch size of 128, a learning 
    rate of 2e-05, and a maximum sequence length of 128. 
    Since this was a classification task, the model was trained with a cross-entropy loss function. 
    The best score the model achieved on this task was 0.8855534709193246, as measured by the 
    eval set accuracy, found after 1 epoch.
    
    For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).
