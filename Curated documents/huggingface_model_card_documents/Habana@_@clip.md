---
license: apache-2.0
---

[Optimum Habana](https://github.com/huggingface/optimum-habana) is the interface between the Hugging Face Transformers and Diffusers libraries and Habana's Gaudi processor (HPU).
It provides a set of tools enabling easy and fast model loading, training and inference on single- and multi-HPU settings for different downstream tasks.
Learn more about how to take advantage of the power of Habana HPUs to train and deploy Transformers and Diffusers models at [hf.co/hardware/habana](https://huggingface.co/hardware/habana).

## CLIP model HPU configuration

This model only contains the `GaudiConfig` file for running CLIP-like models (e.g. [this one](https://huggingface.co/openai/clip-vit-large-patch14)) on Habana's Gaudi processors (HPU).

**This model contains no model weights, only a GaudiConfig.**

This enables to specify:
- `use_fused_adam`: whether to use Habana's custom AdamW implementation
- `use_fused_clip_norm`: whether to use Habana's fused gradient norm clipping operator
- `use_torch_autocast`: whether to use Torch Autocast for managing mixed precision

## Usage

The model is instantiated the same way as in the Transformers library.
The only difference is that there are a few new training arguments specific to HPUs.\
It is strongly recommended to train this model doing bf16 mixed-precision training for optimal performance and accuracy.

[Here](https://github.com/huggingface/optimum-habana/blob/main/examples/contrastive-image-text) is an example script to fine-tune a model on COCO.
Use it as follows:

1. You first need to download the dataset:
```bash
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
cd ..
```

2. Then, you can create a model from pretrained vision and text decoder models:
```python
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-large-patch14", "roberta-large"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")
```

3. Finally, you can run it with the following command:
```bash
python run_clip.py \
    --output_dir ./clip-roberta-finetuned \
    --model_name_or_path ./clip-roberta \
    --data_dir $PWD/data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs \
    --gaudi_config_name Habana/clip \
    --throughput_warmup_steps 2 \
    --bf16
```

Check the [documentation](https://huggingface.co/docs/optimum/habana/index) out for more advanced usage and examples.
