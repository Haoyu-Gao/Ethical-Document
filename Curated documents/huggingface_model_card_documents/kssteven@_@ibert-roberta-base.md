---
{}
---
# I-BERT base model

This model, `ibert-roberta-base`, is an integer-only quantized version of [RoBERTa](https://arxiv.org/abs/1907.11692), and was introduced in [this paper](https://arxiv.org/abs/2101.01321).
I-BERT stores all parameters with INT8 representation, and carries out the entire inference using integer-only arithmetic.
In particular, I-BERT replaces all floating point operations in the Transformer architectures (e.g., MatMul, GELU, Softmax, and LayerNorm) with closely approximating integer operations.
This can result in upto 4x inference speed up as compared to floating point counterpart when tested on an Nvidia T4 GPU.
The best model parameters searched via quantization-aware finetuning can be then exported (e.g., to TensorRT) for integer-only deployment of the model.


## Finetuning Procedure

Finetuning of I-BERT consists of 3 stages: (1) Full-precision finetuning from the pretrained model on a down-stream task, (2) model quantization, and (3) integer-only finetuning (i.e., quantization-aware training) of the quantized model.


### Full-precision finetuning

Full-precision finetuning of I-BERT is similar to RoBERTa finetuning.
For instance, you can run the following command to finetune on the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) text classification task.

```
python examples/text-classification/run_glue.py \
         --model_name_or_path kssteven/ibert-roberta-base \
         --task_name MRPC \
         --do_eval \
         --do_train \
         --evaluation_strategy epoch \
         --max_seq_length 128 \
         --per_device_train_batch_size 32 \
         --save_steps 115 \
         --learning_rate 2e-5 \
         --num_train_epochs 10 \
         --output_dir $OUTPUT_DIR
```

### Model Quantization

Once you are done with full-precision finetuning, open up `config.json` in your checkpoint directory and set the `quantize` attribute as `true`.

```
{                                  
  "_name_or_path": "kssteven/ibert-roberta-base",       
  "architectures": [               
    "IBertForSequenceClassification"                    
  ],                               
  "attention_probs_dropout_prob": 0.1,                  
  "bos_token_id": 0,               
  "eos_token_id": 2,               
  "finetuning_task": "mrpc",       
  "force_dequant": "none",         
  "hidden_act": "gelu",            
  "hidden_dropout_prob": 0.1,      
  "hidden_size": 768,              
  "initializer_range": 0.02,       
  "intermediate_size": 3072,       
  "layer_norm_eps": 1e-05,         
  "max_position_embeddings": 514,  
  "model_type": "ibert",           
  "num_attention_heads": 12,       
  "num_hidden_layers": 12,         
  "pad_token_id": 1,               
  "position_embedding_type": "absolute",                
  "quant_mode": true,             
  "tokenizer_class": "RobertaTokenizer",                
  "transformers_version": "4.4.0.dev0",                 
  "type_vocab_size": 1,            
  "vocab_size": 50265              
}                   
```

Then, your model will automatically run as the integer-only mode when you load the checkpoint.
Also, make sure to delete `optimizer.pt`, `scheduler.pt` and `trainer_state.json` in the same directory.
Otherwise, HF will not reset the optimizer, scheduler, or trainer state for the following integer-only finetuning.


### Integer-only finetuning (Quantization-aware training)

Finally, you will be able to run integer-only finetuning simply by loading the checkpoint file you modified.
Note that the only difference in the example command below is `model_name_or_path`.

```
python examples/text-classification/run_glue.py \
         --model_name_or_path $CHECKPOINT_DIR
         --task_name MRPC \
         --do_eval \
         --do_train \
         --evaluation_strategy epoch \
         --max_seq_length 128 \
         --per_device_train_batch_size 32 \
         --save_steps 115 \
         --learning_rate 1e-6 \
         --num_train_epochs 10 \
         --output_dir $OUTPUT_DIR
```


## Citation info

If you use I-BERT, please cite [our papaer](https://arxiv.org/abs/2101.01321).

```
@article{kim2021bert,
  title={I-BERT: Integer-only BERT Quantization},
  author={Kim, Sehoon and Gholami, Amir and Yao, Zhewei and Mahoney, Michael W and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2101.01321},
  year={2021}
}
```
