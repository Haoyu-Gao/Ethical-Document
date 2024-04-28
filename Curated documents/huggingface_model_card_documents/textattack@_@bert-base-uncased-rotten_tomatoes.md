---
{}
---
## bert-base-uncased fine-tuned with TextAttack on the rotten_tomatoes dataset
    
    This `bert-base-uncased` model was fine-tuned for sequence classificationusing TextAttack 
    and the rotten_tomatoes dataset loaded using the `nlp` library. The model was fine-tuned 
    for 10 epochs with a batch size of 64, a learning 
    rate of 5e-05, and a maximum sequence length of 128. 
    Since this was a classification task, the model was trained with a cross-entropy loss function. 
    The best score the model achieved on this task was 0.875234521575985, as measured by the 
    eval set accuracy, found after 4 epochs.
    
    For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).
