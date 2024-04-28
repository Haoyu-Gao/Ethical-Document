---
license: apache-2.0
---

[Optimum Habana](https://github.com/huggingface/optimum-habana) is the interface between the Hugging Face Transformers and Diffusers libraries and Habana's Gaudi processor (HPU).
It provides a set of tools enabling easy and fast model loading, training and inference on single- and multi-HPU settings for different downstream tasks.
Learn more about how to take advantage of the power of Habana HPUs to train and deploy Transformers and Diffusers models at [hf.co/hardware/habana](https://huggingface.co/hardware/habana).

## T5 model HPU configuration

This model only contains the `GaudiConfig` file for running the [T5](https://huggingface.co/t5-base) model on Habana's Gaudi processors (HPU).

**This model contains no model weights, only a GaudiConfig.**

This enables to specify:
- `use_fused_adam`: whether to use Habana's custom AdamW implementation
- `use_fused_clip_norm`: whether to use Habana's fused gradient norm clipping operator

## Usage

The model is instantiated the same way as in the Transformers library.
The only difference is that there are a few new training arguments specific to HPUs.

[Here](https://github.com/huggingface/optimum-habana/blob/main/examples/summarization/run_summarization.py) is a summarization example script to fine-tune a model. You can run it with T5-small with the following command:
```bash
python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --save_strategy epoch \
    --throughput_warmup_steps 3
```

Check the [documentation](https://huggingface.co/docs/optimum/habana/index) out for more advanced usage and examples. 
